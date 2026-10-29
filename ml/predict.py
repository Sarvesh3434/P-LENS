"""Frozen GRU inference. Output is a stress forecast only — never ADD/PRUNE/REWIRE.

``predict_stress`` returns the class label, class probabilities, the probability
of the attack class, and a normalised entropy in [0, 1].  The controller uses
the probability (not just the arg-max) for load forecasting and dampens
adaptation when the entropy is high.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch

from ml.model import StressGRU
from ml.preprocessing import STRESS_LABELS, normalised_entropy


def _decode(prob: np.ndarray, horizon: int) -> dict:
    k = int(prob.argmax())
    attack_prob = float(prob[STRESS_LABELS.index("Attack")]) if "Attack" in STRESS_LABELS else float(prob[-1])
    return {
        "label": STRESS_LABELS[k],
        "class": k,
        "prob": prob.tolist(),
        "attack_prob": attack_prob,
        "uncertainty": normalised_entropy(prob),
        "horizon": int(horizon),
    }


class StressPredictor:
    def __init__(self, model_path: Path, scaler_path: Path):
        blob = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = StressGRU(
            blob["n_features"],
            blob["hidden"],
            blob["layers"],
            blob["dropout"],
            n_classes=blob.get("n_classes", len(STRESS_LABELS)),
        )
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        self.scaler = joblib.load(scaler_path)
        self.L = blob["L"]
        self.horizon = int(blob.get("horizon", 1))
        self.feature_names = blob.get("feature_names")

    def predict_stress(self, window_hist_scaled: np.ndarray) -> dict:
        """window_hist_scaled: (L, F) already scaled. [ML]"""
        x = torch.from_numpy(window_hist_scaled.astype(np.float32)[None, ...])
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, 1).numpy()[0]
        return _decode(prob, self.horizon)


class TrainedPredictor:
    """Adapter around a just-trained StressGRU so it satisfies the same
    .predict_stress() interface the controller expects from StressPredictor
    (without a round-trip through disk). [ML][ASSUMPTION: duck typing]"""

    def __init__(self, model: StressGRU, scaler, window_length: int, horizon: int = 1):
        self.model = model
        self.model.eval()
        self.scaler = scaler
        self.L = window_length
        self.horizon = int(horizon)

    def predict_stress(self, window_hist_scaled: np.ndarray) -> dict:
        x = torch.from_numpy(window_hist_scaled.astype(np.float32)[None, ...])
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, 1).numpy()[0]
        return _decode(prob, self.horizon)
