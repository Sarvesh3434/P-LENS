"""[DIAGRAM] Plasticity Pressure PP_i(t) per node. [ASSUMPTION] weighted mix of D,F,C,R."""

from __future__ import annotations

from typing import Dict

import numpy as np

from experiments.config import Config


def plasticity_pressure(
    indicators: Dict[str, np.ndarray],
    cfg: Config,
    prev: np.ndarray | None = None,
) -> np.ndarray:
    w = cfg.pressure_weights
    s = w["D"] + w["F"] + w["C"] + w["R"]
    raw = (
        w["D"] * indicators["D"]
        + w["F"] * indicators["F"]
        + w["C"] * indicators["C"]
        + w["R"] * indicators["R"]
    ) / max(s, 1e-9)
    raw = np.clip(raw, 0.0, 1.0)
    if prev is not None and cfg.pressure_ema > 0:
        raw = cfg.pressure_ema * raw + (1.0 - cfg.pressure_ema) * prev
    return np.clip(raw, 0.0, 1.0)
