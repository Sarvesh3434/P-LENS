"""[SIM] Traffic-pattern schedules used by the scenario study.

The demand *matrix* is normalised by :func:`network.traffic.scale_demand`, so
multiplying a matrix by a scalar cancels out.  Patterns therefore modulate the
per-step offered-load target (``traffic_intensity``); spatial demand remains the
dataset/synthetic matrix.  This keeps ``no_ml`` and ``plens`` under identical
traffic and makes the pattern factor observable by the controller.
"""

from __future__ import annotations

import numpy as np

from experiments.config import Config


def replay_series(bundle, steps: int):
    """Held-out demand matrices and features for a controller replay.

    The simulation blocks are disjoint from the ML test blocks and are
    concatenated across every source day, so the replay contains both benign
    and attack regimes.  The concatenated segment is repeated when a run is
    longer than the held-out data.
    """
    if not bundle:
        raise ValueError("replay_series requires a prepared data bundle")
    blocks = bundle.get("split_blocks", {}).get("sim")
    if not blocks:
        lo, hi = bundle["splits"]["sim"]
        blocks = [(lo, hi)]
    dem = np.concatenate([bundle["demand"][a:b] for a, b in blocks if b > a], axis=0)
    scaled = np.concatenate([bundle["scaled"][a:b] for a, b in blocks if b > a], axis=0)
    if len(dem) == 0:
        raise ValueError("empty simulation split")
    reps = int(np.ceil(steps / len(dem)))
    dem = np.concatenate([dem] * reps, axis=0)[:steps]
    scaled = np.concatenate([scaled] * reps, axis=0)[:steps]
    return dem, scaled


def intensity_series(cfg: Config, steps: int) -> np.ndarray:
    """Return a length-``steps`` offered-load intensity trajectory in [0.2, 1.2]."""
    base = float(cfg.traffic_intensity)
    pattern = cfg.traffic_pattern
    t = np.arange(steps, dtype=float)
    if pattern == "shift":
        # Regime change halfway through the run: quiet -> stressed.
        frac = t / max(steps - 1, 1)
        factor = 0.70 + 0.60 * frac
    elif pattern == "periodic":
        period = max(8.0, steps / 4.0)
        factor = 1.0 + 0.35 * np.sin(2.0 * np.pi * t / period)
    elif pattern == "bursty":
        rng = np.random.default_rng(cfg.seed + 90210)
        factor = np.ones(steps, dtype=float)
        burst = 0.0
        for i in range(steps):
            if rng.random() < 0.08:
                burst = 0.9
            else:
                burst *= 0.72
            factor[i] = 0.85 + burst
    else:  # stationary
        factor = np.ones(steps, dtype=float)
    return np.clip(base * factor, 0.20, 1.20)
