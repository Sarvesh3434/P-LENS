"""Plasticity debt as *unresolved structural inadequacy*, not raw pressure.

Why this is not just exponential smoothing
------------------------------------------
``PD_i(t)`` is the bounded accumulated cost of a node remaining structurally
inadequate.  It differs from an EMA / integral-error accumulator in four ways:

1. **Deadband (severity).** Only pressure above a tolerance ``theta`` counts,
   so ordinary load does not inflate debt.
2. **Persistence gate.** A short traffic spike raises ``raw`` for one step but
   barely raises the persistence evidence ``pi_i``; only sustained stress
   contributes materially.  This distinguishes a spike from a regime shift.
3. **Duration and criticality.** The contribution is multiplied by a saturating
   duration factor ``d/(d+tau)`` and by the node's critical-flow exposure, so a
   long-standing problem on a heavily used node accrues more debt than a
   transient excursion on a peripheral node.
4. **Anti-windup saturation.** The increment is proportional to ``(1-PD)`` and
   the leak to ``(1-u)PD``, so debt is bounded in ``[0,1]`` without relying on
   a post-hoc clip, and relief can only reduce it.

Update (all terms in [0,1])::

    raw_i   = clip((PP_i - theta) / (1 - theta), 0, 1)
    pi_i    = beta_p * pi_i + (1 - beta_p) * 1[raw_i > 0]
    d_i     = d_i + 1 if raw_i > 0 else max(0, d_i - 1)
    g_i     = d_i / (d_i + tau)
    u_i     = raw_i * pi_i * g_i * c_i                  (unresolved stress)
    PD_i'   = PD_i + eta * u_i * (1 - PD_i)
                     - gamma * (1 - u_i) * PD_i
                     - relief_i + failure_i

A successful mutation with counterfactual gain ``G`` supplies relief
``delta_r * (1 + tanh(G+)) / 2`` on its closed one-hop neighbourhood.  An
unsuccessful mutation adds a small ``failure_i`` term, so repeated failures do
not leave the controller with the same debt as an untried action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from experiments.config import Config


@dataclass
class DebtState:
    pd: np.ndarray
    persistence: np.ndarray
    duration: np.ndarray

    @classmethod
    def zeros(cls, n: int) -> "DebtState":
        return cls(
            pd=np.zeros(n),
            persistence=np.zeros(n),
            duration=np.zeros(n),
        )


def criticality_from_exposure(net, floor: float = 0.5) -> np.ndarray:
    """Node criticality = demand exposure, normalised to [floor, 1].

    ``exposure_i`` is the share of total demand whose origin or destination is
    ``i``.  A node carrying a large share of the offered load is structurally
    more critical than a peripheral node with the same instantaneous pressure.
    """
    total = float(net.demand.sum()) + 1e-12
    row = net.demand.sum(axis=1)
    col = net.demand.sum(axis=0)
    exposure = (row + col) / total
    exposure = exposure / max(float(exposure.max()), 1e-12)
    return np.clip(floor + (1.0 - floor) * exposure, floor, 1.0)


def update_debt(
    state: DebtState,
    pp: np.ndarray,
    relief: np.ndarray,
    criticality: np.ndarray,
    cfg: Config,
    enabled: bool = True,
    failure: np.ndarray | None = None,
) -> DebtState:
    if not enabled:
        return DebtState.zeros(len(pp))
    theta = float(cfg.debt_tolerance)
    raw = np.clip((pp - theta) / max(1.0 - theta, 1e-9), 0.0, 1.0)
    active = (raw > 0.0).astype(float)
    persistence = np.clip(
        cfg.debt_persist_beta * state.persistence + (1.0 - cfg.debt_persist_beta) * active,
        0.0,
        1.0,
    )
    duration = np.where(raw > 0.0, state.duration + 1.0, np.maximum(0.0, state.duration - 1.0))
    duration = np.minimum(duration, 1e6)
    g = duration / (duration + max(float(cfg.debt_duration_tau), 1e-9))
    crit = np.clip(criticality, cfg.criticality_floor, 1.0)
    unresolved = np.clip(raw * persistence * g * crit, 0.0, 1.0)

    fail = np.zeros_like(pp) if failure is None else np.asarray(failure, dtype=float)
    pd = (
        state.pd
        + cfg.debt_accumulation * unresolved * (1.0 - state.pd)
        - cfg.debt_decay * (1.0 - unresolved) * state.pd
        - np.asarray(relief, dtype=float)
        + fail
    )
    pd = np.clip(pd, 0.0, 1.0)
    return DebtState(pd=pd, persistence=persistence, duration=duration)


def relief_for_mutation(
    n: int,
    nodes_touched: Iterable[int],
    cfg: Config,
    success: bool,
    gain: float = 0.0,
    confidence: float = 1.0,
) -> np.ndarray:
    """Relief on a successful mutation's closed one-hop neighbourhood.

    Relief grows with the confidence-weighted counterfactual gain; a failed
    mutation supplies no relief (the caller adds the failure penalty).
    """
    r = np.zeros(n)
    if not success:
        return r
    scale = 0.5 * (1.0 + float(np.tanh(max(gain, 0.0))))
    amount = cfg.debt_relief * scale * float(np.clip(confidence, 0.0, 1.0))
    for i in nodes_touched:
        if 0 <= i < n:
            r[i] = max(r[i], amount)
    return r


def failure_penalty(n: int, nodes_touched: Iterable[int], cfg: Config, success: bool) -> np.ndarray:
    """Small debt increment for an unsuccessful mutation (bounded by clip)."""
    f = np.zeros(n)
    if success:
        return f
    for i in nodes_touched:
        if 0 <= i < n:
            f[i] = cfg.debt_failure_penalty
    return f
