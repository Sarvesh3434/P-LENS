"""[ML] Frozen inference. Output is stress class/score only — never ADD/PRUNE/REWIRE."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch

from ml.model import StressGRU
from ml.preprocessing import STRESS_LABELS


class StressPredictor:
    def __init__(self, model_path: Path, scaler_path: Path):
        blob = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = StressGRU(blob["n_features"], blob["hidden"], blob["layers"], blob["dropout"])
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        self.scaler = joblib.load(scaler_path)
        self.L = blob["L"]

    def predict_stress(self, window_hist_scaled: np.ndarray) -> dict:
        """window_hist_scaled: (L, F) already scaled. [ML]"""
        x = torch.from_numpy(window_hist_scaled.astype(np.float32)[None, ...])
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, 1).numpy()[0]
        k = int(prob.argmax())
        return {"label": STRESS_LABELS[k], "class": k, "prob": prob.tolist()}


class TrainedPredictor:
    """Adapter around a just-trained StressGRU so it satisfies the same
    .predict_stress() interface the controller expects from StressPredictor
    (without a round-trip through disk). [ML][ASSUMPTION: duck typing]"""

    def __init__(self, model: StressGRU, scaler, window_length: int):
        self.model = model
        self.model.eval()
        self.scaler = scaler
        self.L = window_length

    def predict_stress(self, window_hist_scaled: np.ndarray) -> dict:
        x = torch.from_numpy(window_hist_scaled.astype(np.float32)[None, ...])
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, 1).numpy()[0]
        k = int(prob.argmax())
        return {"label": STRESS_LABELS[k], "class": k, "prob": prob.tolist()}
