"""[ML] Leakage-free test metrics and conventional baselines.

Reported quantities: accuracy, macro precision/recall/F1, per-class
precision/recall/F1/support, confusion matrix, ROC-AUC, Brier score, temporal
drift slices, persistence, majority class, a traffic-composite threshold rule,
logistic regression and a random forest on the same chronological split.
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
        pred = prob.argmax(1)
    return pred, prob


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


def permutation_importance(model, X, y, n_repeats: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    base_pred, _ = _predict(model, X)
    base = f1_score(y, base_pred, average="macro", zero_division=0)
    n_feat = X.shape[-1]
    drops = []
    for f in range(n_feat):
        ds = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, :, f] = Xp[perm, :, f]
            p, _ = _predict(model, Xp)
            ds.append(base - f1_score(y, p, average="macro", zero_division=0))
        drops.append(float(np.mean(ds)))
    return {"base_macro_f1": float(base), "drop": drops}


def _baseline_scores(bundle) -> dict:
    """Conventional baselines trained on the same chronological split."""
    Xtr, ytr, _ = bundle["seq"]["train"]
    Xte, yte, idx = bundle["seq"]["test"]
    y_all = bundle["y"]
    prev = np.array([y_all[i - 1] if i - 1 >= 0 else y_all[i] for i in idx])

    out = {
        "persistence": _score(yte, prev),
        "majority": _score(yte, np.full_like(yte, int(np.bincount(ytr, minlength=2).argmax()))),
    }

    # A simple traffic-only threshold rule (no learned weights).
    tr_feat = bundle["feats"][bundle["splits"]["train"][0] : bundle["splits"]["train"][1]]
    te_feat = bundle["feats"][bundle["splits"]["test"][0] : bundle["splits"]["test"][1]]
    thr = float(np.median(stress_composite(tr_feat)))
    # sequences shorten the test slice; align the rule to the sequence targets
    comp = stress_composite(te_feat)
    rule = np.zeros(len(yte), dtype=np.int64)
    for n, row in enumerate(idx):
        local = int(row - bundle["splits"]["test"][0])
        if 0 <= local < len(comp) and comp[local] >= thr:
            rule[n] = 1
    out["traffic_threshold"] = _score(yte, rule)

    flat_tr = Xtr.reshape(len(Xtr), -1)
    flat_te = Xte.reshape(len(Xte), -1)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(flat_tr, ytr)
    out["logistic_regression"] = _score(yte, lr.predict(flat_te), lr.predict_proba(flat_te))

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=0,
        n_jobs=-1,
    )
    rf.fit(flat_tr, ytr)
    out["random_forest"] = _score(yte, rf.predict(flat_te), rf.predict_proba(flat_te))
    return out


def _drift_slices(y, pred, prob, n_slices: int = 3) -> list[dict]:
    """Per-tercile test metrics: the test segment is chronological."""
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


def evaluate_split(model, X, y, y_seq_prev=None) -> dict:
    pred, prob = _predict(model, X)
    out = _score(y, pred, prob)
    out["report"] = classification_report(
        y, pred, target_names=STRESS_LABELS, zero_division=0
    )
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
    return out


def run_evaluation(cfg: Config, model, bundle) -> dict:
    Xte, yte, idx = bundle["seq"]["test"]
    y_all = bundle["y"]
    prev = np.array([y_all[i - 1] if i - 1 >= 0 else y_all[i] for i in idx])
    metrics = evaluate_split(model, Xte, yte, prev)
    pred, prob = _predict(model, Xte)
    metrics["drift_slices"] = _drift_slices(yte, pred, prob)
    metrics["permutation_importance"] = permutation_importance(model, Xte, yte)
    metrics["permutation_features"] = bundle["feature_names"]
    metrics["baselines"] = _baseline_scores(bundle)
    metrics["horizon"] = int(bundle.get("horizon", 1))
    metrics["leakage_free"] = bool(bundle.get("leakage_free", False))
    metrics["n_test"] = int(len(yte))

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
                "beats_persistence": metrics["beats_persistence"],
                "persistence_macro_f1": metrics["persistence_macro_f1"],
                "majority_macro_f1": metrics["majority_macro_f1"],
                "horizon": metrics["horizon"],
                "leakage_free": metrics["leakage_free"],
                "baselines": metrics["baselines"],
            },
            indent=2,
        )
    )
    return metrics
