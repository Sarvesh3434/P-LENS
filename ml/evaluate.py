"""[ML] Test-set metrics vs persistence and majority baselines."""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from experiments.config import Config
from ml.preprocessing import STRESS_LABELS


def _predict(model, X) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)))
        prob = torch.softmax(logits, 1).numpy()
        pred = prob.argmax(1)
    return pred, prob


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


def evaluate_split(model, X, y, y_seq_prev=None) -> dict:
    pred, prob = _predict(model, X)
    acc = accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()
    auc = None
    try:
        auc = float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    except Exception:
        auc = None
    out = {
        "accuracy": float(acc),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "confusion_matrix": cm,
        "roc_auc_ovr": auc,
        "report": classification_report(y, pred, target_names=STRESS_LABELS, zero_division=0),
    }
    # persistence: predict next = last window's class (use previous target if aligned)
    if y_seq_prev is not None and len(y_seq_prev) == len(y):
        pers = y_seq_prev
    else:
        pers = np.r_[y[:1], y[:-1]]
    out["persistence_macro_f1"] = float(f1_score(y, pers, average="macro", zero_division=0))
    out["persistence_accuracy"] = float(accuracy_score(y, pers))
    maj = np.full_like(y, int(np.bincount(y, minlength=3).argmax()))
    out["majority_macro_f1"] = float(f1_score(y, maj, average="macro", zero_division=0))
    out["majority_accuracy"] = float(accuracy_score(y, maj))
    out["beats_persistence"] = bool(out["macro_f1"] >= out["persistence_macro_f1"])
    return out


def run_evaluation(cfg: Config, model, bundle) -> dict:
    Xte, yte, idx = bundle["seq"]["test"]
    y_all = bundle["y"]
    prev = np.array([y_all[i - 1] if i - 1 >= 0 else y_all[i] for i in idx])
    metrics = evaluate_split(model, Xte, yte, prev)
    metrics["permutation_importance"] = permutation_importance(model, Xte, yte)
    metrics["permutation_features"] = bundle["feature_names"]
    (cfg.artifacts_dir() / "ml_metrics.json").write_text(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))
    (cfg.artifacts_dir() / "ml_report.txt").write_text(metrics["report"] + "\n" + json.dumps({
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "beats_persistence": metrics["beats_persistence"],
        "persistence_macro_f1": metrics["persistence_macro_f1"],
        "majority_macro_f1": metrics["majority_macro_f1"],
    }, indent=2))
    return metrics
