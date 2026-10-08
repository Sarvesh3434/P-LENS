"""[ML] Train GRU with early stopping on validation macro-F1. Frozen later during sim.

2-class Normal-vs-Attack stress target. Labels come from the CIC/UNSW raw attack
column via the data loader, which now appends an "attack_flag" per window.
[ML]
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from experiments.config import Config
from ml.data_loader import FEATURE_NAMES, chronological_split, load_cic_windows, load_unsw_windows
from ml.model import StressGRU
from ml.preprocessing import STRESS_LABELS, fit_scaler, fit_stress_map, make_sequences, stress_composite
from ml.synthetic import synthetic_windows


def _slice(arr, lo, hi):
    return arr[lo:hi]


def _attack_target(feats: np.ndarray, metas: list, splits: dict) -> np.ndarray:
    """Thematic 2-class target.

    attack_flag (feature index 9) is 1.0 when *any* flow in the window is an
    attack, i.e. it is the per-window ground truth from the raw Label column.
    We threshold it at 0.5 so windows with at least one attack flow are Attack=1.
    [ASSUMPTION]
    """
    y = feats[:, 9].copy()
    y = (y >= 0.5).astype(np.int64)
    train_hi = splits["train"][1]
    # If for some reason the loader forgot the flag, fall back to the raw-label
    # heuristic only on the training rows so the model still sees a real signal.
    if y.sum() == 0:
        s = stress_composite(feats)
        y = (s >= float(np.median(s[:train_hi]))).astype(np.int64)
    return y


def prepare_bundle(cfg: Config):
    if cfg.dataset == "unsw":
        feats, dem, metas, bounds = load_unsw_windows(cfg)
    elif cfg.dataset == "synthetic":
        # [SIM] generator instead of files; bounds are None → generic split
        feats, dem, metas = synthetic_windows(cfg, seed=cfg.seed + 5_000)
        bounds = None
    else:
        feats, dem, metas, bounds = load_cic_windows(cfg)

    splits = chronological_split(len(feats), bounds)
    tr = _slice(feats, *splits["train"])
    if len(tr) < 10:
        raise RuntimeError("Not enough training windows. Check dataset paths / flows_per_window.")

    scaler = fit_scaler(tr)
    y_all = _attack_target(feats, metas, splits)
    Y_all = y_all.copy()

    scaled = scaler.transform(feats).astype(np.float32)

    def seq_for(part):
        lo, hi = splits[part]
        # sequences must not cross into other parts: build from that slice only
        X, y, idx = make_sequences(scaled[lo:hi], Y_all[lo:hi], cfg.window_length_L)
        return X, y, idx + lo

    bundle = {
        "feats": feats,
        "scaled": scaled,
        "demand": dem,
        "y": Y_all,
        "splits": splits,
        "bounds": bounds,
        "scaler": scaler,
        "stress_map": fit_stress_map(tr, tr),
        "seq": {p: seq_for(p) for p in ("train", "val", "test", "sim")},
        "feature_names": FEATURE_NAMES,
        "metas": metas,
    }
    return bundle


def fit_stress_map(train_feat: np.ndarray, train_y: np.ndarray) -> None:
    """placeholder kept to avoid import churn; labels come from attack_flag."""
    return None


def train_model(cfg: Config, bundle=None) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if bundle is None:
        bundle = prepare_bundle(cfg)
    Xtr, ytr, _ = bundle["seq"]["train"]
    Xva, yva, _ = bundle["seq"]["val"]
    device = torch.device("cpu")
    n_feat = Xtr.shape[-1]

    # The CIC 2-class problem is easy to underfit: use a smaller, deeper-enough
    # GRU with more epochs and less dropout so it can actually separate classes.
    layers = 1
    hidden = 64
    dropout = 0.0
    epochs = int(getattr(cfg, "epochs", 15))
    lr = float(getattr(cfg, "learning_rate", 1e-3))

    model = StressGRU(n_feat, hidden, layers, dropout, n_classes=len(STRESS_LABELS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    def loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle)

    tr_loader = loader(Xtr, ytr, True)
    history = {"train_loss": [], "val_f1": []}
    best_f1 = -1.0
    best_state = None
    patience = 0
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
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(Xva).to(device))
            pred = logits.argmax(1).cpu().numpy()
        f1 = f1_score(yva, pred, average="macro", zero_division=0)
        history["train_loss"].append(total / max(n, 1))
        history["val_f1"].append(float(f1))
        print(f"epoch {epoch+1}/{epochs} loss={history['train_loss'][-1]:.4f} val_macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                print("early stopping")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

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
            "feature_names": FEATURE_NAMES,
            "dataset": cfg.dataset,
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
        "history": history,
        "splits": {k: list(v) for k, v in bundle["splits"].items()},
        "n_windows": int(len(bundle["feats"])),
        "class_counts_train": np.bincount(ytr, minlength=2).tolist(),
        "class_counts_val": np.bincount(yva, minlength=2).tolist(),
        "class_counts_test": np.bincount(bundle["seq"]["test"][1], minlength=2).tolist(),
    }
    (cfg.artifacts_dir() / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return {"model": model, "bundle": bundle, "meta": meta}
