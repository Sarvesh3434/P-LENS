"""[ML] Train the GRU stress forecaster with a leakage-free target.

The model sees only the nine pre-label traffic features defined in
``ml.data_loader.FEATURE_NAMES``.  The binary attack/normal target is supplied
in a separate array.  Preprocessing is fit on the training split only, the
split is chronological, and early stopping uses validation macro-F1.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from experiments.config import Config
from ml.data_loader import (
    FEATURE_NAMES,
    chronological_split,
    day_blocked_blocks,
    load_cic_windows,
    load_unsw_windows,
)
from ml.model import StressGRU
from ml.preprocessing import STRESS_LABELS, fit_scaler, fit_stress_map, make_sequences
from ml.synthetic import synthetic_windows


def _slice(arr, lo, hi):
    return arr[lo:hi]


def prepare_bundle(cfg: Config):
    if cfg.dataset == "unsw":
        feats, dem, labels, metas, bounds = load_unsw_windows(cfg)
    elif cfg.dataset == "synthetic":
        feats, dem, labels, metas = synthetic_windows(cfg, seed=cfg.seed + 5_000)
        bounds = None
    else:
        feats, dem, labels, metas, bounds = load_cic_windows(cfg)

    if labels.shape[0] != feats.shape[0]:
        raise RuntimeError("labels and features are misaligned; loader bug")
    if feats.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(
            f"feature matrix has {feats.shape[1]} columns but FEATURE_NAMES has "
            f"{len(FEATURE_NAMES)}: the target may have leaked into the inputs"
        )

    splits = chronological_split(len(feats), bounds)
    split_blocks = day_blocked_blocks(len(feats), bounds)
    tr_parts = [feats[a:b] for a, b in split_blocks["train"]]
    tr = np.concatenate(tr_parts, axis=0)
    if len(tr) < 10:
        raise RuntimeError("Not enough training windows. Check dataset paths / flows_per_window.")

    scaler = fit_scaler(tr)
    scaled = scaler.transform(feats).astype(np.float32)

    def seq_for(part):
        xs, ys, idx = [], [], []
        for lo, hi in split_blocks[part]:
            # sequences never cross a block boundary, so no future window from
            # another day can enter the input history
            X, y, i = make_sequences(
                scaled[lo:hi], labels[lo:hi], cfg.window_length_L, cfg.ml_horizon
            )
            if len(X):
                xs.append(X)
                ys.append(y)
                idx.append(i + lo)
        if not xs:
            return (
                np.zeros((0, cfg.window_length_L, feats.shape[1])),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
            )
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(idx)

    bundle = {
        "feats": feats,
        "scaled": scaled,
        "demand": dem,
        "labels": labels,
        "y": labels,
        "splits": splits,
        "split_blocks": split_blocks,
        "split_protocol": "day-blocked chronological (65/15/15/5 per source block)",
        "scaler": scaler,
        "stress_map": fit_stress_map(tr, labels[: len(tr)]),
        "seq": {p: seq_for(p) for p in ("train", "val", "test", "sim")},
        "feature_names": FEATURE_NAMES,
        "metas": metas,
        "horizon": cfg.ml_horizon,
        "leakage_free": True,
    }
    return bundle


def train_model(cfg: Config, bundle=None) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if bundle is None:
        bundle = prepare_bundle(cfg)
    Xtr, ytr, _ = bundle["seq"]["train"]
    Xva, yva, _ = bundle["seq"]["val"]
    device = torch.device("cpu")
    n_feat = Xtr.shape[-1]

    hidden = int(getattr(cfg, "gru_hidden", 64))
    layers = int(getattr(cfg, "gru_layers", 2))
    dropout = float(getattr(cfg, "dropout", 0.2)) if layers > 1 else 0.0
    epochs = int(getattr(cfg, "epochs", 15))
    lr = float(getattr(cfg, "learning_rate", 1e-3))

    model = StressGRU(n_feat, hidden, layers, dropout, n_classes=len(STRESS_LABELS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    counts = np.bincount(ytr, minlength=len(STRESS_LABELS)).astype(float)
    if getattr(cfg, "use_class_weights", True) and (counts > 0).all():
        weights = counts.sum() / (len(counts) * counts)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32)
        )
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    def loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle)

    tr_loader = loader(Xtr, ytr, True)
    history = {"train_loss": [], "val_f1": [], "val_threshold": []}
    best_f1 = -1.0
    best_state = None
    best_threshold = 0.5
    patience = 0
    grid = np.linspace(0.05, 0.95, 91)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            prob = torch.softmax(model(torch.from_numpy(Xva).to(device)), 1)[:, 1].cpu().numpy()
        # Early stopping uses the *thresholded* validation macro-F1, the same
        # criterion reported on test; argmax alone is biased by class weights.
        th_scores = [f1_score(yva, (prob >= t).astype(int), average="macro", zero_division=0) for t in grid]
        k = int(np.argmax(th_scores))
        f1 = float(th_scores[k])
        thr = float(grid[k])
        history["train_loss"].append(total / max(n, 1))
        history["val_f1"].append(f1)
        history["val_threshold"].append(thr)
        print(
            f"epoch {epoch+1}/{epochs} loss={history['train_loss'][-1]:.4f} "
            f"val_macro_f1={f1:.4f} (thr={thr:.2f})"
        )
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_threshold = thr
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                print("early stopping")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    decision_threshold = float(best_threshold)
    print(f"decision threshold (val macro-F1) = {decision_threshold:.2f}")

    cfg.model_path().parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": n_feat,
            "hidden": hidden,
            "layers": layers,
            "dropout": dropout,
            "n_classes": len(STRESS_LABELS),
            "L": cfg.window_length_L,
            "horizon": cfg.ml_horizon,
            "feature_names": FEATURE_NAMES,
            "dataset": cfg.dataset,
            "decision_threshold": decision_threshold,
            "leakage_free": True,
        },
        cfg.model_path(),
    )
    joblib.dump(bundle["scaler"], cfg.scaler_path())
    np.savez_compressed(
        cfg.processed_path(),
        scaled=bundle["scaled"],
        y=bundle["y"],
        demand=bundle["demand"],
        feats=bundle["feats"],
    )
    meta = {
        "best_val_macro_f1": best_f1,
        "decision_threshold": decision_threshold,
        "history": history,
        "horizon": cfg.ml_horizon,
        "leakage_free": True,
        "feature_names": FEATURE_NAMES,
        "split_protocol": bundle["split_protocol"],
        "splits": {k: list(v) for k, v in bundle["splits"].items()},
        "split_blocks": {k: [list(b) for b in v] for k, v in bundle["split_blocks"].items()},
        "n_windows": int(len(bundle["feats"])),
        "class_counts_train": np.bincount(ytr, minlength=2).tolist(),
        "class_counts_val": np.bincount(yva, minlength=2).tolist(),
        "class_counts_test": np.bincount(bundle["seq"]["test"][1], minlength=2).tolist(),
    }
    (cfg.artifacts_dir() / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return {"model": model, "bundle": bundle, "meta": meta, "threshold": decision_threshold}
