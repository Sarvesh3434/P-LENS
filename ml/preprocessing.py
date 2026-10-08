"""[ML] Scaling (train-only fit), stress labels from CIC/UNSW raw attack labels.

Two-class target [ML]: an attack window is anything the dataset marks as not
benign/normal. This is thematic (matches the attack_cat / Label column) and
separated, unlike the old ECDF-tertile split which forced 3 overlapping classes
out of a single scalar composite. [ASSUMPTION]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


STRESS_LABELS = ["Normal", "Attack"]  # 2-class stress target


def stress_composite(feat: np.ndarray) -> np.ndarray:
    """packet rate, byte rate, new-flow rate (window flow_count). [ML][ASSUMPTION]"""
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


def fit_stress_map(
    train_y: np.ndarray, train_feat: np.ndarray
) -> StressMap:
    """Threshold the composite stress at its median over the *training* split.

    The composite is a *ranking* signal used only to bootstrap a fixed threshold
    so the model sees a roughly balanced 2-class target during training.
    [ASSUMPTION]
    """
    s = stress_composite(train_feat)
    thr = float(np.median(s))
    return StressMap(threshold=thr)


def make_sequences(feat: np.ndarray, y: np.ndarray, L: int):
    """X[t] = feat[t-L:t], target = y[t] (next-window stress of the window after the history).
    We predict stress of window t from windows t-L ... t-1, so target index = t, length L history.
    """
    xs, ys, idx = [], [], []
    for t in range(L, len(feat)):
        xs.append(feat[t - L : t])
        ys.append(y[t])
        idx.append(t)
    if not xs:
        return np.zeros((0, L, feat.shape[1])), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.stack(xs), np.array(ys, dtype=np.int64), np.array(idx, dtype=np.int64)


def fit_scaler(train_feat: np.ndarray) -> StandardScaler:
    sc = StandardScaler()
    sc.fit(train_feat)
    return sc
