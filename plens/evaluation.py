"""Counterfactual mutation evaluation with matched traffic and failures.

Two estimators are provided.

1. ``replay_counterfactual`` (default).  At the mutation step we snapshot the
   complete pre-failure world.  At evaluation time we replay two branches from
   that snapshot for exactly ``k`` steps:

   * **factual**: all logged mutations, including the target;
   * **counterfactual**: the same mutations *except* the target.

   Both branches consume the identical logged demand, the identical failure
   random-number state, and the identical concurrent mutations.  The only
   difference is the target mutation.  This separates its effect from concurrent
   topology changes, failures and traffic variation.  Uncertainty is estimated
   by replaying the pair under ``B`` matched demand perturbations (lognormal OD
   noise, renormalised to the original total) and forming a lower confidence
   bound on the gain.  Credit is only updated when the effect exceeds
   ``cf_min_effect`` and the estimate is confident enough.

2. ``shadow`` (legacy).  Reverse the mutation on the *current* topology.  This
   is cheap but confounded by everything that happened in between; it is kept
   as a comparison/ablation, not as the default.

``before_after`` is also kept as a naive estimator for comparison.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple

import numpy as np

from network.failures import apply_failures
from network.graph import EvolvingNetwork, canon
from network.traffic import route_and_load
from plens import mutations as muts

# EdgeState objects saved before a legacy shadow revert, keyed by canon edge [SIM]
_shadow_saved: Dict[Tuple[int, int], object] = {}


def delayed_gain(
    real_metrics: Dict[str, float],
    shadow_metrics: Dict[str, float],
    cfg,
    n_mutations: int = 0,
) -> Dict[str, float]:
    """Gain of the factual topology relative to the counterfactual one."""
    w = cfg.gain_weights
    lat_scale = max(real_metrics.get("latency", 0.0), shadow_metrics.get("latency", 0.0), 1e-9)
    util = (
        (shadow_metrics.get("latency", 0.0) - real_metrics.get("latency", 0.0)) / lat_scale
        + (shadow_metrics.get("loss", 0.0) - real_metrics.get("loss", 0.0))
        + (shadow_metrics.get("congestion", 0.0) - real_metrics.get("congestion", 0.0))
        + (real_metrics.get("throughput", 0.0) - shadow_metrics.get("throughput", 0.0))
    ) / 4.0
    res = real_metrics.get("connected", 0.0) - shadow_metrics.get("connected", 0.0)
    res += (real_metrics.get("n_edges", 0.0) - shadow_metrics.get("n_edges", 0.0)) / max(cfg.network_size, 1)
    stab = abs(real_metrics.get("avg_path_length", 0.0) - shadow_metrics.get("avg_path_length", 0.0))
    if not np.isfinite(stab):
        stab = 0.0
    # Churn is a small tie-breaker, not a dominant cost: a single edge change
    # costs at most 5% of the [0,1] stability term.  Without this scaling every
    # mutation would look harmful on a well-connected 24-node graph.
    if cfg.max_mutations_per_step > 0 and n_mutations > 0:
        stab += 0.05 * n_mutations / cfg.max_mutations_per_step
    stab = min(stab, 1.0)
    g = w["u"] * util + w["r"] * res - w["s"] * stab
    return {
        "utility": float(util),
        "resilience": float(res),
        "stability_cost": float(stab),
        "G": float(g),
    }


def _perturb_demand(d: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return d
    noise = rng.lognormal(mean=0.0, sigma=sigma, size=d.shape)
    np.fill_diagonal(noise, 0.0)
    out = d * noise
    total = float(d.sum())
    s = float(out.sum())
    if s > 0 and total > 0:
        out *= total / s
    return out


def _apply_logged(branch: EvolvingNetwork, cand: dict) -> bool:
    """Apply a logged candidate to a replay branch; skip if infeasible."""
    ok, _ = muts.apply(branch, cand)
    return ok


def replay_branch(
    snapshot: EvolvingNetwork,
    logs: Sequence[dict],
    horizon: int,
    exclude_token: int | None,
    cfg,
    demand_override: Sequence[np.ndarray] | None = None,
) -> Dict[str, float]:
    """Replay one branch from ``snapshot`` for ``horizon`` steps.

    ``logs[s]`` must contain ``rng_state`` (before failures), ``demand`` (the
    scaled matrix used by the real step), and ``mutations`` (candidate dicts
    with a ``token``).  ``logs`` is indexed relative to the replay start.
    """
    branch = copy.deepcopy(snapshot)
    trajectory = []
    for s in range(horizon):
        entry = logs[s]
        branch.step = int(entry.get("t", branch.step + 1))
        branch.rng.bit_generator.state = copy.deepcopy(entry["rng_state"])
        apply_failures(branch)
        d = entry["demand"] if demand_override is None else demand_override[s]
        branch.set_demand(np.asarray(d, dtype=float))
        route_and_load(branch)
        trajectory.append(dict(branch.last_metrics))
        for item in entry.get("mutations", []):
            if exclude_token is not None and item.get("token") == exclude_token:
                continue
            _apply_logged(branch, item["cand"])
    # metrics after each routing pass, before the next step's mutations
    return trajectory


def evaluate_replay(
    snapshot: EvolvingNetwork,
    logs: Sequence[dict],
    horizon: int,
    token: int,
    cfg,
    n_mutations: int = 1,
) -> Dict[str, float]:
    real_traj = replay_branch(snapshot, logs[:horizon], horizon, None, cfg)
    shadow_traj = replay_branch(snapshot, logs[:horizon], horizon, token, cfg)
    ev = delayed_gain(real_traj[-1], shadow_traj[-1], cfg, n_mutations)

    b = int(getattr(cfg, "cf_bootstrap", 0))
    if b <= 0:
        # Cheap matched uncertainty: the horizon steps are paired observations
        # of the same intervention.  Use their dispersion instead of extra
        # replays; this keeps the default estimator fast and still conservative.
        step_gains = [delayed_gain(r, s, cfg, n_mutations)["G"] for r, s in zip(real_traj, shadow_traj)]
        std = float(np.std(step_gains, ddof=1) / np.sqrt(len(step_gains))) if len(step_gains) > 1 else 0.0
        lcb = float(ev["G"] - cfg.cf_z * std)
        ev.update(
            {
                "gain_std": std,
                "gain_lcb": lcb,
                "confidence": float(1.0 / (1.0 + std)),
                "bootstrap": 0,
                "step_gains": step_gains,
            }
        )
        return ev

    rng = np.random.default_rng(10_000 + int(token))
    gains = []
    for _ in range(b):
        demand_override = [
            _perturb_demand(np.asarray(entry["demand"], dtype=float), rng, cfg.cf_perturb_sigma)
            for entry in logs[:horizon]
        ]
        r = replay_branch(snapshot, logs[:horizon], horizon, None, cfg, demand_override)[-1]
        c = replay_branch(snapshot, logs[:horizon], horizon, token, cfg, demand_override)[-1]
        gains.append(delayed_gain(r, c, cfg, n_mutations)["G"])
    std = float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0
    lcb = float(ev["G"] - cfg.cf_z * std / np.sqrt(max(len(gains), 1)))
    confidence = float(1.0 / (1.0 + std))
    ev.update(
        {
            "gain_std": std,
            "gain_lcb": lcb,
            "confidence": confidence,
            "bootstrap": len(gains),
            "bootstrap_gains": gains,
        }
    )
    return ev


def credit_is_valid(ev: Dict[str, float], cfg) -> bool:
    """Gate credit updates on effect size and uncertainty."""
    if ev.get("G", 0.0) == 0.0 and ev.get("gain_std", 0.0) == 0.0:
        return False
    return (
        abs(float(ev.get("gain_lcb", ev.get("G", 0.0)))) >= cfg.cf_min_effect
        and float(ev.get("confidence", 1.0)) >= cfg.cf_min_confidence
    )


def shadow_eval(net: EvolvingNetwork, detail: dict, cfg, n_mutations: int = 0) -> Dict[str, float]:
    """Legacy: temporarily reverse the mutation on the current topology. [ASSUMPTION]"""
    real = dict(net.last_metrics)
    _shadow_saved.clear()
    if detail.get("removed"):
        key = canon(*detail["removed"])
        st = net.edges.get(key)
        if st is not None:
            _shadow_saved[key] = st
    muts.revert_one(net, detail)
    shadow = route_and_load(net)
    if detail.get("added"):
        net.add_edge(*detail["added"])
    if detail.get("removed"):
        a, b = detail["removed"]
        if not net.G.has_edge(a, b):
            net.G.add_edge(a, b)
            key = canon(a, b)
            if key not in net.edges:
                saved = _shadow_saved.get(key)
                if saved is not None:
                    net.edges[key] = saved
    route_and_load(net)
    return delayed_gain(real, shadow, cfg, n_mutations)


def evaluate_pending(net: EvolvingNetwork, detail: dict, cfg, n_mutations: int = 0) -> Dict[str, float]:
    """Legacy dispatch used by tests and the ``shadow``/``before_after`` modes."""
    if cfg.eval_mode == "shadow":
        return shadow_eval(net, detail, cfg, n_mutations)
    before = detail.get("before") or net.last_metrics
    ev = delayed_gain(net.last_metrics, before, cfg, n_mutations)
    ev["mode"] = "before_after"
    return ev

