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
    tr = _slice(feats, *splits["train"])
    if len(tr) < 10:
        raise RuntimeError("Not enough training windows. Check dataset paths / flows_per_window.")

    scaler = fit_scaler(tr)
    scaled = scaler.transform(feats).astype(np.float32)

    def seq_for(part):
        lo, hi = splits[part]
        # sequences never cross into other parts: build from that slice only
        X, y, idx = make_sequences(
            scaled[lo:hi], labels[lo:hi], cfg.window_length_L, cfg.ml_horizon
        )
        return X, y, idx + lo

    bundle = {
        "feats": feats,
        "scaled": scaled,
        "demand": dem,
        "labels": labels,
        "y": labels,
        "splits": splits,
        "bounds": bounds,
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            "horizon": cfg.ml_horizon,
            "feature_names": FEATURE_NAMES,
            "dataset": cfg.dataset,
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
        "history": history,
        "horizon": cfg.ml_horizon,
        "leakage_free": True,
        "feature_names": FEATURE_NAMES,
        "splits": {k: list(v) for k, v in bundle["splits"].items()},
        "n_windows": int(len(bundle["feats"])),
        "class_counts_train": np.bincount(ytr, minlength=2).tolist(),
        "class_counts_val": np.bincount(yva, minlength=2).tolist(),
        "class_counts_test": np.bincount(bundle["seq"]["test"][1], minlength=2).tolist(),
    }
    (cfg.artifacts_dir() / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return {"model": model, "bundle": bundle, "meta": meta}
