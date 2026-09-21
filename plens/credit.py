"""[DIAGRAM] Mutation credit MC_ij from delayed gain. [ASSUMPTION] feeds next MF."""

from __future__ import annotations

from typing import Dict, Tuple

import math

from network.graph import canon


def update_credit(
    mc: Dict[Tuple[int, int], float],
    i: int,
    j: int,
    gain: float,
    lam: float,
) -> float:
    key = canon(i, j)
    new = math.tanh(gain)
    old = mc.get(key, 0.0)
    val = lam * old + (1.0 - lam) * new
    mc[key] = val
    return val
