"""Dependency-aware plasticity debt as accumulated structural maladaptation.

The model separates **temporary network stress** from **persistent structural
inadequacy** and treats debt as the bounded accumulation of the latter, not as
an integral of raw constraint violations.

Definitions (all per node ``i`` and step ``t``, all quantities in ``[0, 1]``)::

    raw_i(t)  = clip((PP_i(t) - theta) / (1 - theta), 0, 1)   # severity
    pi_i(t)   = beta * pi_i(t-1) + (1-beta) * 1[raw_i(t) > 0] # persistence
    d_i(t)    = d_i(t-1) + 1 if raw_i > 0 else max(0, d_i-1)  # duration
    g_i(t)    = d_i(t) / (d_i(t) + tau)                        # saturating age
    u_i(t)    = raw_i * pi_i * g_i * c_i                       # local inadequacy

Dependency propagation.  Let ``P`` be a row-stochastic dependency matrix built
from the usable topology, neighbour criticality and inter-node demand coupling.
An overloaded node raises the inadequacy of the nodes that depend on it::

    q_i(t)    = sum_j P_ij * max(0, u_j(t) - rho)              # propagated part
    s_i(t)    = clip(u_i(t) + kappa * q_i(t), 0, 1)            # structural inadequacy

Rate term.  Structural maladaptation has a direction.  A worsening node accrues
debt faster; an improving node releases it::

    r_i(t)    = s_i(t) - s_i(t-1)                              # inadequacy velocity

Modification suppression.  While a node's neighbourhood is being reconfigured,
its transient inadequacy is partly explained by the in-flight topology change.
A gate ``m_i(t) in [0,1]`` (1 = no modification in flight) damps accumulation so
the controller does not double-count a stress it is already repairing::

    m_i(t)    = 1 / (1 + mu * inflight_i(t))

Bounded update with explicit projection/saturation operators::

    drive_i   = eta * s_i * (1 - PD_i) + eta_r+ * max(0, r_i) * (1 - PD_i)
    leak_i    = gamma * (1 - s_i) * PD_i + eta_r- * max(0, -r_i) * PD_i
    PD_i'     = Proj_[0,1]( PD_i + m_i * drive_i - leak_i
                            - Sat_{[0, PD_i + m_i drive_i]}(relief_i)
                            + Sat_{[0, 1 - (...)]}(failure_i) )

Why this is not an integral controller
--------------------------------------
* Only *excess* pressure above a deadband enters, and only after persistence
  evidence and duration have accumulated.
* Temporary stress ``raw`` and persistent inadequacy ``s`` are separate state
  variables; a one-step spike cannot create material debt.
* The velocity term makes the update respond to the *direction* of structural
  change, not just its level.
* Neighbour inadequacy is propagated through a topology/demand dependency
  matrix, so a single overloaded node raises the debt of its dependants.
* Topology modification suppresses accumulation in the affected neighbourhood.
* The update is a projected monotone map.  For constant ``s`` and no mutations
  the fixed point is ``PD* = eta*s / (eta*s + gamma*(1-s))`` and the map is a
  contraction with factor ``|1 - eta*s - gamma*(1-s)| < 1`` whenever
  ``0 < eta, gamma < 1`` (see :mod:`plens.theory`).

A successful mutation with confidence-weighted counterfactual gain ``G`` supplies
relief on its closed one-hop neighbourhood; relief is diffused through the same
dependency matrix.  An unsuccessful mutation adds a bounded failure term, so
repeated failures do not leave the controller with the same debt as an untried
action.

Every formula above is an explicitly documented model assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from experiments.config import Config


# --------------------------------------------------------------------------- state
@dataclass
class DebtState:
    pd: np.ndarray
    persistence: np.ndarray
    duration: np.ndarray
    # New state variables.  Defaults keep the public constructor backwards
    # compatible with the original three-field DebtState.
    inadequacy: np.ndarray | None = None
    rate: np.ndarray | None = None
    propagated: np.ndarray | None = None
    stress: np.ndarray | None = None

    @classmethod
    def zeros(cls, n: int) -> "DebtState":
        return cls(
            pd=np.zeros(n),
            persistence=np.zeros(n),
            duration=np.zeros(n),
            inadequacy=np.zeros(n),
            rate=np.zeros(n),
            propagated=np.zeros(n),
            stress=np.zeros(n),
        )


# ----------------------------------------------------------------------- operators
def project_interval(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Explicit Euclidean projection onto ``[lo, hi]``."""
    return np.clip(np.asarray(x, dtype=float), lo, hi)


def saturate(x: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Saturating operator ``Sat_{[0, upper]}(x)`` used for relief/failure."""
    x = np.asarray(x, dtype=float)
    upper = np.maximum(np.asarray(upper, dtype=float), 0.0)
    return np.minimum(np.maximum(x, 0.0), upper)


def _ablation_flags(cfg: Config) -> dict:
    a = {x.strip() for x in str(getattr(cfg, "ablation", "none")).split(",") if x.strip()}
    return {
        "persistence": "no_persistence" not in a,
        "criticality": "no_criticality" not in a,
        "propagation": "no_debt_propagation" not in a,
        "rate": "no_debt_rate" not in a,
    }


# --------------------------------------------------------------------- dependency
def criticality_from_exposure(net, floor: float = 0.5) -> np.ndarray:
    """Node criticality = demand exposure, normalised to ``[floor, 1]``.

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


def dependency_matrix(net, criticality: np.ndarray, cfg: Config) -> np.ndarray:
    """Row-stochastic dependency matrix ``P`` on the usable topology.

    ``P_ij`` is the share of node ``i``'s dependency that flows to neighbour
    ``j``.  The weight combines neighbour criticality with the relative demand
    coupling between the two nodes, so a heavily used neighbour that also
    exchanges traffic with ``i`` propagates more of its inadequacy.

    Complexity: ``O(n + m)`` for the topology plus ``O(m)`` demand lookups.
    """
    n = int(net.n)
    P = np.zeros((n, n), dtype=float)
    total_dem = float(net.demand.sum()) + 1e-12
    alive = set(net.alive_nodes())
    coupling = float(getattr(cfg, "debt_dependency_coupling", 0.0))
    for i in range(n):
        if i not in alive:
            continue
        for j in net.neighbours(i):
            if j not in alive:
                continue
            flow = (float(net.demand[j, i]) + float(net.demand[i, j])) / total_dem
            P[i, j] = max(float(criticality[j]), 0.0) * (1.0 + coupling * n * flow)
    row = P.sum(axis=1, keepdims=True)
    return np.divide(P, row, out=np.zeros_like(P), where=row > 1e-12)


def propagate_inadequacy(
    local: np.ndarray,
    dependency: np.ndarray | None,
    cfg: Config,
    tolerance: float | None = None,
    weight: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(propagated, combined)`` inadequacy arrays."""
    local = np.clip(np.asarray(local, dtype=float), 0.0, 1.0)
    if dependency is None:
        return np.zeros_like(local), local
    rho = float(getattr(cfg, "debt_propagation_tolerance", 0.0) if tolerance is None else tolerance)
    kappa = float(getattr(cfg, "debt_propagation", 0.0) if weight is None else weight)
    q = np.asarray(dependency, dtype=float) @ np.maximum(local - rho, 0.0)
    combined = np.clip(local + kappa * q, 0.0, 1.0)
    return q, combined


def modification_gate(
    n: int,
    recent_mutations: Sequence,
    t: int,
    cfg: Config,
) -> np.ndarray:
    """Suppress debt accumulation near topology modifications in flight.

    ``recent_mutations`` may contain ``(t0, nodes)`` pairs or dictionaries with
    keys ``t``/``t0`` and ``nodes``.  Events older than ``debt_mod_window`` are
    ignored; within the window the influence decays exponentially.  The gate is
    always in ``(0, 1]`` and equals 1 away from any modification.
    """
    gate = np.ones(n, dtype=float)
    window = max(int(getattr(cfg, "debt_mod_window", 1)), 1)
    strength = float(getattr(cfg, "debt_mod_suppression", 0.0))
    if strength <= 0 or not recent_mutations:
        return gate
    inflight = np.zeros(n, dtype=float)
    for ev in recent_mutations:
        if isinstance(ev, dict):
            t0 = int(ev.get("t", ev.get("t0", t)))
            nodes = ev.get("nodes", ())
        else:
            t0, nodes = int(ev[0]), ev[1]
        age = int(t) - t0
        if age < 0 or age > window:
            continue
        decay = float(np.exp(-age / window))
        for node in nodes:
            node = int(node)
            if 0 <= node < n:
                inflight[node] += decay
    return 1.0 / (1.0 + strength * inflight)


def propagate_relief(relief: np.ndarray, dependency: np.ndarray | None, cfg: Config) -> np.ndarray:
    """Diffuse mutation relief to dependent neighbours (bounded by debt_relief)."""
    relief = np.asarray(relief, dtype=float)
    if dependency is None:
        return relief
    kappa = float(getattr(cfg, "debt_relief_propagation", 0.0))
    if kappa <= 0:
        return relief
    spread = kappa * (np.asarray(dependency, dtype=float) @ relief)
    return np.clip(relief + spread, 0.0, float(cfg.debt_relief))


# ------------------------------------------------------------------------- update
def update_debt(
    state: DebtState,
    pp: np.ndarray,
    relief: np.ndarray,
    criticality: np.ndarray,
    cfg: Config,
    enabled: bool = True,
    failure: np.ndarray | None = None,
    dependency: np.ndarray | None = None,
    modification_gate: np.ndarray | None = None,
) -> DebtState:
    """Advance the debt state by one step.

    Backwards compatible with the original call signature; the new optional
    ``dependency`` and ``modification_gate`` arguments activate propagation and
    modification suppression respectively.
    """
    if not enabled:
        return DebtState.zeros(len(pp))

    flags = _ablation_flags(cfg)
    n = len(pp)
    pp = np.asarray(pp, dtype=float)
    theta = float(cfg.debt_tolerance)
    raw = np.clip((pp - theta) / max(1.0 - theta, 1e-9), 0.0, 1.0)
    active = (raw > 0.0).astype(float)

    if flags["persistence"]:
        persistence = np.clip(
            cfg.debt_persist_beta * state.persistence
            + (1.0 - cfg.debt_persist_beta) * active,
            0.0,
            1.0,
        )
    else:
        persistence = active

    duration = np.where(
        raw > 0.0, state.duration + 1.0, np.maximum(0.0, state.duration - 1.0)
    )
    duration = np.minimum(duration, 1e6)
    g = duration / (duration + max(float(cfg.debt_duration_tau), 1e-9))
    crit = np.clip(criticality, cfg.criticality_floor, 1.0) if flags["criticality"] else np.ones(n)

    local = np.clip(raw * persistence * g * crit, 0.0, 1.0)

    if flags["propagation"] and dependency is not None:
        q, s = propagate_inadequacy(local, dependency, cfg)
    else:
        q = np.zeros(n, dtype=float)
        s = local

    prev_s = state.inadequacy
    if prev_s is None:
        prev_s = np.zeros(n, dtype=float)
    prev_s = np.asarray(prev_s, dtype=float)
    rate = s - prev_s

    if flags["rate"]:
        eta_up = float(getattr(cfg, "debt_rate_gain", 0.0))
        eta_down = float(getattr(cfg, "debt_rate_decay", 0.0))
    else:
        eta_up = eta_down = 0.0

    pd = np.clip(np.asarray(state.pd, dtype=float), 0.0, 1.0)
    gate = np.ones(n, dtype=float) if modification_gate is None else np.clip(modification_gate, 0.0, 1.0)

    drive = (
        float(cfg.debt_accumulation) * s * (1.0 - pd)
        + eta_up * np.maximum(rate, 0.0) * (1.0 - pd)
    )
    leak = (
        float(cfg.debt_decay) * (1.0 - s) * pd
        + eta_down * np.maximum(-rate, 0.0) * pd
    )

    # Saturation operators keep relief/failure inside the available interval.
    upper_relief = pd + gate * drive
    relief_sat = saturate(np.asarray(relief, dtype=float), upper_relief)
    fail = np.zeros_like(pd) if failure is None else np.asarray(failure, dtype=float)
    pre_failure = pd + gate * drive - leak - relief_sat
    failure_sat = saturate(fail, 1.0 - pre_failure)
    pd_new = project_interval(pre_failure + failure_sat, 0.0, 1.0)

    return DebtState(
        pd=pd_new,
        persistence=persistence,
        duration=duration,
        inadequacy=s,
        rate=rate,
        propagated=q,
        stress=raw,
    )


def relief_potential(pd: np.ndarray) -> np.ndarray:
    """Headroom for relief: ``PD * (1 - PD)``.

    This is deliberately *not* ``PD`` itself.  Current debt already enters the
    mutation score through the debt term; the relief potential measures how much
    of that debt one mutation can plausibly remove.  It is maximal at moderate
    debt and vanishes for a saturated node (which needs structural repair, not a
    single edge change) and for an unstressed node (which has nothing to relieve).
    """
    pd = np.clip(np.asarray(pd, dtype=float), 0.0, 1.0)
    return pd * (1.0 - pd)


def debt_fixed_point(s: float, cfg: Config) -> float:
    """Stable point of the no-mutation, constant-inadequacy map."""
    eta = float(cfg.debt_accumulation)
    gamma = float(cfg.debt_decay)
    s = float(np.clip(s, 0.0, 1.0))
    denom = eta * s + gamma * (1.0 - s)
    return float(eta * s / denom) if denom > 1e-12 else 0.0


def debt_contraction_factor(s: float, cfg: Config) -> float:
    """Local contraction factor of the no-mutation debt map at inadequacy ``s``."""
    eta = float(cfg.debt_accumulation)
    gamma = float(cfg.debt_decay)
    s = float(np.clip(s, 0.0, 1.0))
    return float(abs(1.0 - eta * s - gamma * (1.0 - s)))


# ------------------------------------------------------------------------ relief
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
