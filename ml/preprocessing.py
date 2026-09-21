"""[ML] Scaling (train-only fit), stress labels from training ECDF tertiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


STRESS_LABELS = ["Low", "Medium", "High"]


def stress_composite(feat: np.ndarray) -> np.ndarray:
    """packet rate, byte rate, new-flow rate (window flow_count). [ML][ASSUMPTION]"""
    flow_count = feat[:, 0]
    pkt_rate = feat[:, 1]
    byte_rate = feat[:, 2]
    return np.log1p(flow_count) + np.log1p(pkt_rate) + np.log1p(byte_rate)


@dataclass
class StressMap:
    tertiles: Tuple[float, float]

    def to_class(self, values: np.ndarray) -> np.ndarray:
        t1, t2 = self.tertiles
        y = np.zeros(len(values), dtype=np.int64)
        y[values >= t1] = 1
        y[values >= t2] = 2
        return y

    def label_name(self, k: int) -> str:
        return STRESS_LABELS[int(k)]


def fit_stress_map(train_feat: np.ndarray) -> StressMap:
    s = stress_composite(train_feat)
    t1, t2 = np.quantile(s, [1.0 / 3.0, 2.0 / 3.0])
    return StressMap((float(t1), float(t2)))


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
