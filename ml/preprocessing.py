"""[ML] Leakage-free scaling and sequence construction.

The supervised target is the binary attack/normal label returned separately by
the data loaders.  No feature is derived from that label.  Sequences use only
windows strictly before the prediction time, and the target can be placed
``horizon`` steps ahead so the controller consumes a genuine forecast rather
than a one-step-lagged label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


STRESS_LABELS = ["Normal", "Attack"]  # 2-class stress target


def stress_composite(feat: np.ndarray) -> np.ndarray:
    """Traffic-only severity score used for diagnostics, never as a label.

    All three inputs are pre-label traffic aggregates, so this is safe to
    compute at prediction time.  It is *not* used to define the supervision
    target in the leakage-free pipeline.
    """
    flow_count = feat[:, 0]
    pkt_rate = feat[:, 1]
    byte_rate = feat[:, 2]
    return np.log1p(flow_count) + np.log1p(pkt_rate) + np.log1p(byte_rate)


@dataclass
class StressMap:
    threshold: float

    def to_class(self, values: np.ndarray) -> np.ndarray:
        return (values >= self.threshold).astype(np.int64)

    def label_name(self, k: int) -> str:
        return STRESS_LABELS[int(k)]


def fit_stress_map(train_feat: np.ndarray, train_y: np.ndarray) -> StressMap:
    """Diagnostic median threshold; the model target remains the real label."""
    s = stress_composite(train_feat)
    return StressMap(threshold=float(np.median(s)))


def make_sequences(
    feat: np.ndarray,
    y: np.ndarray,
    L: int,
    horizon: int = 1,
):
    """X[t] = feat[t-L:t], target = y[t+horizon-1].

    ``horizon`` is the forecast lead.  History never crosses the segment
    boundary because callers pass only the slice belonging to one split.
    """
    horizon = max(1, int(horizon))
    xs, ys, idx = [], [], []
    row = 0
    for t in range(L, len(feat)):
        target = t + horizon - 1
        if target >= len(feat):
            break
        xs.append(feat[t - L : t])
        ys.append(y[target])
        idx.append(target)
        row += 1
    if not xs:
        return (
            np.zeros((0, L, feat.shape[1])),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    return np.stack(xs), np.array(ys, dtype=np.int64), np.array(idx, dtype=np.int64)


def fit_scaler(train_feat: np.ndarray) -> StandardScaler:
    """Fit scaling parameters on the training split only."""
    sc = StandardScaler()
    sc.fit(train_feat)
    return sc


def normalised_entropy(prob: np.ndarray) -> float:
    """Entropy of a class-probability vector in [0, 1] (0 = certain)."""
    p = np.asarray(prob, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    h = -float(np.sum(p * np.log(p)))
    return float(h / np.log(max(len(p), 2))) if len(p) > 1 else 0.0
