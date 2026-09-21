"""[DIAGRAM] Delayed performance evaluation at t+k.

Default mode `counterfactual` [ASSUMPTION]: compare the real topology against the
same topology with the mutation reversed (shadow), on identical demand. This is a
simulation-only oracle — a real network cannot undo history — and is documented
as such. Alternative `before_after` [ASSUMPTION]: compare the metrics recorded at
mutation time (stored in `detail["before"]`) with the current metrics; it is
biased by traffic drift between t and t+k, which is exactly why counterfactual is
the default.

Gain [ASSUMPTION] (all terms in [0, 1]):
    G = w_u * UtilityGain + w_r * ResilienceGain - w_s * StabilityCost
    UtilityGain    = mean(dLatency_norm, dLoss, dCongestion, dThroughput)
    ResilienceGain = dConnected + delta_edges / network_size (alternate-path proxy)
    StabilityCost  = |dAvgPathLength| + churn, with churn = n this step /
                   (n + max): the mutated share of the step's total allowance
                   (n made + n still allowed), so bookkeeping damps but cannot
                   veto a strictly-improving mutation [ASSUMPTION]
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from network.graph import Edge, EdgeState, EvolvingNetwork, canon
from network.traffic import route_and_load
from plens import mutations as muts

# EdgeState objects saved before a shadow revert, keyed by canon edge [SIM]
_shadow_saved: Dict[Edge, "EdgeState"] = {}


def delayed_gain(
    real_metrics: Dict[str, float],
    shadow_metrics: Dict[str, float],
    cfg,
    n_mutations: int = 0,
) -> Dict[str, float]:
    """Counterfactual gain: real G_{t+k} vs shadow G_{t+k} without the mutation."""
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
    # churn is measured as the mutated share of the step's total allowance
    # (n made + n still allowed); the no-op case (n=0) is free [ASSUMPTION]
    if cfg.max_mutations_per_step > 0 and n_mutations > 0:
        stab += n_mutations / (n_mutations + cfg.max_mutations_per_step)
    stab = min(stab, 1.0)
    g = w["u"] * util + w["r"] * res - w["s"] * stab
    return {
        "utility": float(util),
        "resilience": float(res),
        "stability_cost": float(stab),
        "G": float(g),
    }


def shadow_eval(net: EvolvingNetwork, detail: dict, cfg, n_mutations: int = 0) -> Dict[str, float]:
    """Temporarily reverse the mutation, re-route identical demand, restore. [ASSUMPTION]"""
    real = dict(net.last_metrics)
    # stash EdgeStates of edges the revert will delete, so the restore can put
    # the exact original objects back [SIM]
    _shadow_saved.clear()
    if detail.get("removed"):
        key = canon(*detail["removed"])
        st = net.edges.get(key)
        if st is not None:
            _shadow_saved[key] = st
    muts.revert_one(net, detail)
    shadow = route_and_load(net)
    # restore the real topology in reverse order (undo the revert), then recompute
    # loads so net.last_metrics matches the real topology again [SIM]
    if detail.get("added"):
        net.add_edge(*detail["added"])
    if detail.get("removed"):
        a, b = detail["removed"]
        # re-add manually so a pre-existing EdgeState (bandwidth, reliability,
        # queue state...) is restored intact instead of deleted by _sync [SIM]
        if not net.G.has_edge(a, b):
            net.G.add_edge(a, b)
            key = canon(a, b)
            if key not in net.edges:
                saved = _shadow_saved.get(key)
                if saved is not None:
                    net.edges[key] = saved
    route_and_load(net)
    return delayed_gain(real, shadow, cfg, n_mutations)


def evaluate_pending(
    net: EvolvingNetwork,
    detail: dict,
    cfg,
    n_mutations: int = 0,
) -> Dict[str, float]:
    """Dispatch on cfg.eval_mode. `before_after` compares metrics frozen at
    mutation time with the metrics now (same demand realisation)."""
    if cfg.eval_mode == "counterfactual":
        return shadow_eval(net, detail, cfg, n_mutations)
    before = detail.get("before") or net.last_metrics
    ev = delayed_gain(net.last_metrics, before, cfg, n_mutations)
    ev["mode"] = "before_after"
    return ev
