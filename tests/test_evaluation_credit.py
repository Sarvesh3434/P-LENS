"""[DIAGRAM] Delayed Performance Evaluation + Mutation Credit (boxes 9–10)."""
from __future__ import annotations

import numpy as np

from network.traffic import route_and_load, scale_demand
from plens.credit import update_credit
from plens.evaluation import delayed_gain, evaluate_pending, shadow_eval


def _m(lat=0.005, loss=0.0, cong=0.4, thr=0.9, conn=1.0, edges=12, apl=6.0):
    return {
        "latency": lat,
        "loss": loss,
        "congestion": cong,
        "throughput": thr,
        "connected": conn,
        "n_edges": float(edges),
        "avg_path_length": apl,
    }


def test_gain_positive_when_mutation_helps(cfg):
    real = _m(lat=0.004, loss=0.0, cong=0.3, thr=1.0, edges=13, apl=5.0)
    shadow = _m(lat=0.005, loss=0.1, cong=0.5, thr=0.9, edges=12, apl=5.0)
    ev = delayed_gain(real, shadow, cfg, n_mutations=1)
    assert ev["G"] > 0.0
    assert ev["utility"] > 0.0 and ev["resilience"] > 0.0
    assert set(ev) == {"utility", "resilience", "stability_cost", "G"}


def test_gain_negative_when_mutation_hurts(cfg):
    real = _m(lat=0.008, loss=0.2, cong=0.7, thr=0.8, edges=12)
    shadow = _m(lat=0.004, loss=0.0, cong=0.3, thr=1.0, edges=12)
    ev = delayed_gain(real, shadow, cfg, n_mutations=0)
    assert ev["G"] < 0.0


def test_stability_cost_includes_churn_and_clips(cfg):
    # churn = n_mutations / max_mutations_per_step, clipped into [0, 1]
    ev = delayed_gain(_m(), _m(), cfg, n_mutations=cfg.max_mutations_per_step)
    assert ev["stability_cost"] <= 1.0
    assert ev["stability_cost"] > 0.0


def test_shadow_eval_reverses_and_restores(net, cfg, demand):
    net.set_demand(scale_demand(demand, net, cfg.traffic_intensity))
    route_and_load(net)
    assert net.add_edge(0, 5)
    detail = {"op": "ADD", "added": [0, 5], "removed": None}
    ev = shadow_eval(net, detail, cfg, n_mutations=1)
    # restored: edge back in place, evaluation is a dict with finite numbers
    assert net.G.has_edge(0, 5)
    assert set(ev) >= {"utility", "resilience", "stability_cost", "G"}
    assert all(np.isfinite(v) for v in ev.values())


def test_shadow_eval_prune_roundtrip(net, cfg, demand):
    net.set_demand(scale_demand(demand, net, cfg.traffic_intensity))
    route_and_load(net)
    n_before = net.G.number_of_edges()
    # (0,1) exists in the ring; pruning must be feasible min-degree-wise? ring degree=2,
    # min_degree=2 would forbid it — bypass validate() and test evaluation machinery only.
    detail = {"op": "PRUNE", "added": None, "removed": [0, 1]}
    ev = shadow_eval(net, detail, cfg, n_mutations=0)
    assert net.G.number_of_edges() == n_before
    assert set(ev) >= {"G"}


def test_evaluate_pending_before_after_mode(net, cfg, demand):
    cfg.eval_mode = "before_after"
    net.set_demand(scale_demand(demand, net, cfg.traffic_intensity))
    route_and_load(net)
    detail = {"op": "ADD", "added": [0, 5], "removed": None, "before": _m()}
    ev = evaluate_pending(net, detail, cfg, n_mutations=1)
    assert ev.get("mode") == "before_after"
    assert set(ev) >= {"utility", "resilience", "stability_cost", "G"}


def test_credit_tanh_ema_and_canonical_key():
    mc = {}
    v1 = update_credit(mc, 5, 0, gain=1.0, lam=0.7)
    assert (0, 5) in mc and v1 == mc[(0, 5)]
    assert np.isclose(v1, (1 - 0.7) * np.tanh(1.0))
    v2 = update_credit(mc, 0, 5, gain=1.0, lam=0.7)
    assert np.isclose(v2, 0.7 * v1 + 0.3 * np.tanh(1.0))


def test_credit_negative_gain_gives_negative_credit():
    mc = {}
    v = update_credit(mc, 1, 2, gain=-2.0, lam=0.5)
    assert v < 0.0
    # tanh bounds the update to (-1, 1)
    assert -1.0 < v < 0.0
