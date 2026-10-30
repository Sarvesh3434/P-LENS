"""[ML] Leakage-free test metrics, calibration, and conventional baselines.

Reported quantities: accuracy, macro precision/recall/F1, per-class metrics,
confusion matrix, ROC-AUC, Brier score, expected calibration error, temporal
drift slices, persistence, majority class, a traffic-composite threshold rule,
logistic regression, and a random forest on the same day-blocked chronological
split.

The split protocol is deliberately strict.  A single global "train on the first
days, test on the last day" split makes the test day contain attack families
that never appear in training (e.g. Friday DDoS/port-scan after Mon--Thu), which
collapses every model to chance.  Instead each contiguous source block (a CIC
day) is split 65/15/15/5 in time order: no test window precedes a training
window from the same block, the scaler is fit on training blocks only, and
sequences never cross block boundaries.  The final 5% of every day is reserved
for the closed-loop controller replay and is disjoint from the ML test blocks.
Persistence is retained as a reference because it is a *label-oracle* baseline
(it sees the previous ground-truth label, which an online detector does not),
so it bounds rather than competes with deployable predictors.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from experiments.config import Config
from ml.preprocessing import STRESS_LABELS, stress_composite


def _predict(model, X) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)))
        prob = torch.softmax(logits, 1).numpy()
    return prob, prob


def _score(y, pred, prob=None) -> dict:
    p, r, f1, support = precision_recall_fscore_support(
        y, pred, labels=list(range(len(STRESS_LABELS))), zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(p[k]),
                "recall": float(r[k]),
                "f1": float(f1[k]),
                "support": int(support[k]),
            }
            for k, name in enumerate(STRESS_LABELS)
        },
        "confusion_matrix": confusion_matrix(
            y, pred, labels=list(range(len(STRESS_LABELS)))
        ).tolist(),
    }
    if prob is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y, prob[:, 1]))
            out["brier"] = float(brier_score_loss(y, prob[:, 1]))
        except Exception:
            out["roc_auc"] = None
            out["brier"] = None
    return out


def expected_calibration_error(y, prob, n_bins: int = 10) -> dict:
    """Binned calibration error for the positive class (lower is better)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        mask = (p >= lo) & (p < hi if k < n_bins - 1 else p <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += float(np.sum(mask)) / len(p) * abs(conf - acc)
        bins.append({"lo": lo, "hi": hi, "n": int(np.sum(mask)), "confidence": conf, "accuracy": acc})
    return {"ece": float(ece), "bins": bins}


def permutation_importance(model, X, y, n_repeats: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    base_prob, _ = _predict(model, X)
    base = f1_score(y, base_prob.argmax(1), average="macro", zero_division=0)
    n_feat = X.shape[-1]
    drops = []
    for f in range(n_feat):
        ds = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, :, f] = Xp[perm, :, f]
            p, _ = _predict(model, Xp)
            ds.append(base - f1_score(y, p.argmax(1), average="macro", zero_division=0))
        drops.append(float(np.mean(ds)))
    return {"base_macro_f1": float(base), "drop": drops}


def _block_indices(bundle, part: str) -> np.ndarray:
    return np.concatenate([np.arange(a, b) for a, b in bundle["split_blocks"][part]])


def _baseline_scores(bundle, threshold: float = 0.5) -> dict:
    """Conventional baselines evaluated on the same chronological split."""
    Xtr, ytr, _ = bundle["seq"]["train"]
    Xte, yte, idx = bundle["seq"]["test"]
    y_all = bundle["y"]
    prev = np.array([y_all[max(i - 1, 0)] for i in idx])

    out = {
        "persistence": _score(yte, prev),
        "majority": _score(yte, np.full_like(yte, int(np.bincount(ytr, minlength=2).argmax()))),
    }

    # A deployable traffic-only threshold rule (no labels in its input).
    tr_idx = _block_indices(bundle, "train")
    thr = float(np.median(stress_composite(bundle["feats"][tr_idx])))
    comp = stress_composite(bundle["feats"][idx])
    out["traffic_threshold"] = _score(yte, (comp >= thr).astype(np.int64))

    flat_tr = Xtr.reshape(len(Xtr), -1)
    flat_te = Xte.reshape(len(Xte), -1)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(flat_tr, ytr)
    lr_prob = lr.predict_proba(flat_te)
    out["logistic_regression"] = _score(yte, (lr_prob[:, 1] >= threshold).astype(int), lr_prob)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=0,
        n_jobs=-1,
    )
    rf.fit(flat_tr, ytr)
    rf_prob = rf.predict_proba(flat_te)
    out["random_forest"] = _score(yte, (rf_prob[:, 1] >= threshold).astype(int), rf_prob)
    return out


def _drift_slices(y, pred, prob, n_slices: int = 3) -> list[dict]:
    """Per-tercile test metrics: the concatenated test blocks stay in time order."""
    n = len(y)
    out = []
    for k in range(n_slices):
        lo = int(k * n / n_slices)
        hi = int((k + 1) * n / n_slices)
        if hi <= lo:
            continue
        out.append(
            {
                "slice": k,
                "start": lo,
                "end": hi,
                **_score(y[lo:hi], pred[lo:hi], prob[lo:hi] if prob is not None else None),
            }
        )
    return out


def evaluate_split(model, X, y, y_seq_prev=None, threshold: float = 0.5) -> dict:
    prob, _ = _predict(model, X)
    pred = (prob[:, 1] >= threshold).astype(np.int64)
    out = _score(y, pred, prob)
    out["report"] = classification_report(
        y, pred, target_names=STRESS_LABELS, zero_division=0
    )
    out["calibration"] = expected_calibration_error(y, prob[:, 1])
    if y_seq_prev is not None and len(y_seq_prev) == len(y):
        pers = y_seq_prev
    else:
        pers = np.r_[y[:1], y[:-1]]
    out["persistence_macro_f1"] = float(
        f1_score(y, pers, average="macro", zero_division=0)
    )
    out["persistence_accuracy"] = float(accuracy_score(y, pers))
    maj = np.full_like(y, int(np.bincount(y, minlength=len(STRESS_LABELS)).argmax()))
    out["majority_macro_f1"] = float(f1_score(y, maj, average="macro", zero_division=0))
    out["majority_accuracy"] = float(accuracy_score(y, maj))
    out["beats_persistence"] = bool(out["macro_f1"] >= out["persistence_macro_f1"])
    out["decision_threshold"] = float(threshold)
    return out


def run_evaluation(cfg: Config, model, bundle, threshold: float | None = None) -> dict:
    Xte, yte, idx = bundle["seq"]["test"]
    y_all = bundle["y"]
    prev = np.array([y_all[max(i - 1, 0)] for i in idx])
    if threshold is None:
        threshold = 0.5
    metrics = evaluate_split(model, Xte, yte, prev, threshold=threshold)
    prob, _ = _predict(model, Xte)
    pred = (prob[:, 1] >= threshold).astype(np.int64)
    metrics["drift_slices"] = _drift_slices(yte, pred, prob)
    metrics["permutation_importance"] = permutation_importance(model, Xte, yte)
    metrics["permutation_features"] = bundle["feature_names"]
    metrics["baselines"] = _baseline_scores(bundle, threshold)
    metrics["horizon"] = int(bundle.get("horizon", 1))
    metrics["leakage_free"] = bool(bundle.get("leakage_free", False))
    metrics["split_protocol"] = bundle.get("split_protocol")
    metrics["split_blocks"] = {
        k: [list(b) for b in v] for k, v in bundle.get("split_blocks", {}).items()
    }
    metrics["n_test"] = int(len(yte))
    metrics["n_train"] = int(len(bundle["seq"]["train"][0]))
    metrics["n_val"] = int(len(bundle["seq"]["val"][0]))

    (cfg.artifacts_dir() / "ml_metrics.json").write_text(
        json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2)
    )
    (cfg.artifacts_dir() / "ml_report.txt").write_text(
        metrics["report"]
        + "\n"
        + json.dumps(
            {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "roc_auc": metrics.get("roc_auc"),
                "brier": metrics.get("brier"),
                "ece": metrics["calibration"]["ece"],
                "beats_persistence": metrics["beats_persistence"],
                "persistence_macro_f1": metrics["persistence_macro_f1"],
                "majority_macro_f1": metrics["majority_macro_f1"],
                "horizon": metrics["horizon"],
                "decision_threshold": metrics["decision_threshold"],
                "leakage_free": metrics["leakage_free"],
                "split_protocol": metrics["split_protocol"],
                "baselines": metrics["baselines"],
            },
            indent=2,
        )
    )
    return metrics
