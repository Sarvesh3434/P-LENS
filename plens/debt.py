"""[DIAGRAM] Plasticity Debt PD_i(t) — accumulated unresolved pressure. [ASSUMPTION]"""

from __future__ import annotations

import numpy as np

from experiments.config import Config


def update_debt(
    pd: np.ndarray,
    pp: np.ndarray,
    relief: np.ndarray,
    cfg: Config,
    enabled: bool = True,
) -> np.ndarray:
    if not enabled:
        return np.zeros_like(pd)
    nxt = cfg.debt_decay * pd + cfg.debt_accumulation * pp - relief
    return np.clip(nxt, 0.0, 1.0)


def relief_for_mutation(n: int, nodes_touched: list[int], cfg: Config, success: bool) -> np.ndarray:
    """[ASSUMPTION] relief on endpoints and 1-hop is applied by the caller for neighbours."""
    r = np.zeros(n)
    if not success:
        return r
    for i in nodes_touched:
        if 0 <= i < n:
            r[i] = cfg.debt_relief
    return r
