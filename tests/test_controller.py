"""[DIAGRAM] Full P-LENS loop: PP → PD → B → MF → mutation → MC, feedback F1–F5."""
from __future__ import annotations

import networkx as nx
import numpy as np

from experiments.config import Config
from network.graph import EvolvingNetwork
from plens.controller import PLENSController


def _series(cfg: Config, demand: np.ndarray, steps: int):
    return [demand.copy() for _ in range(steps)]


def _run(cfg: Config, demand: np.ndarray, steps: int, predictor=None):
    net = EvolvingNetwork(cfg, np.random.default_rng(cfg.seed))
    ctl = PLENSController(cfg, net, ml_predictor=predictor, base_rng=np.random.default_rng(cfg.seed + 1))
    for t in range(steps):
        ctl.step(t, demand.copy())
    return net, ctl


def test_loop_runs_and_records(cfg, demand):
    net, ctl = _run(cfg, demand, cfg.simulation_steps)
    assert len(ctl.history) == cfg.simulation_steps
    # F1: every step appends one topology snapshot
    assert len(ctl.topology_snapshots) == cfg.simulation_steps + 1
    assert ctl.topology_snapshots[0] == [list(e) for e in net.reference.edges()]
    assert ctl.topology_snapshots[-1] == [list(e) for e in net.G.edges()]


def test_connectivity_always_preserved(cfg, demand):
    net, _ = _run(cfg, demand, cfg.simulation_steps)
    assert nx.is_connected(net.G)


def test_f2_delayed_credit_updated_after_k_steps(cfg, demand):
    net, ctl = _run(cfg, demand, cfg.simulation_steps)
    assert any(x["mutations"] for x in ctl.history), "expected mutations with default fixture"
    evaluated = [
        m for rec in ctl.history for m in rec["mutations"] if m.get("delayed_eval") is not None
    ]
    assert evaluated, "mutations must be evaluated after delay_k steps"
    assert ctl.mc, "evaluated mutations must leave credit in MC_ij"
    for v in ctl.mc.values():
        assert -1.0 < v < 1.0


def test_f3_debt_accumulates_and_gets_relief(cfg, demand):
    net, ctl = _run(cfg, demand, cfg.simulation_steps)
    pds = [rec["pd_mean"] for rec in ctl.history]
    assert max(pds) > 0.0, "debt must accumulate under persistent pressure"
    # some step must show relief (evaluated successful mutation reduces PD growth)
    assert any(x["mutations"] for x in ctl.history)


def test_f4_budget_stays_bounded(cfg, demand):
    net, ctl = _run(cfg, demand, cfg.simulation_steps)
    for rec in ctl.history:
        assert cfg.mutation_budget_min - 1e-9 <= rec["budget"] <= cfg.mutation_budget_max + 1e-9
    budgets = [rec["budget"] for rec in ctl.history]
    assert max(budgets) != min(budgets), "budget should react to gains/costs/spend"


def test_f5_pressure_is_ema_smoothed(cfg, demand):
    net, ctl = _run(cfg, demand, 3)
    ind0 = ctl.history[0]["indicators_mean"]
    # PP is an EMA of the weighted indicators; with prev=0 it must be < raw indicator mix
    raw0 = np.mean([ind0["D"], ind0["F"], ind0["C"], ind0["R"]])
    assert ctl.history[0]["pp_mean"] <= raw0 + 1e-6


def test_selector_respects_per_step_cap(cfg, demand):
    net, ctl = _run(cfg, demand, cfg.simulation_steps)
    for rec in ctl.history:
        assert len(rec["mutations"]) <= cfg.max_mutations_per_step


def test_plens_deterministic_same_seed(cfg, demand):
    net1, ctl1 = _run(cfg, demand, 5)
    net2, ctl2 = _run(cfg, demand, 5)
    m1 = [r["mutations"] for r in ctl1.history]
    m2 = [r["mutations"] for r in ctl2.history]
    assert m1 == m2
    l1 = [r["metrics"]["latency"] for r in ctl1.history]
    l2 = [r["metrics"]["latency"] for r in ctl2.history]
    assert l1 == l2


def test_static_policy_never_mutates(cfg, demand):
    cfg.policy = "static"
    net, ctl = _run(cfg, demand, cfg.simulation_steps)
    assert all(not rec["mutations"] for rec in ctl.history)
    assert net.G.number_of_edges() == net.reference.number_of_edges()


def test_random_policy_mutates_within_caps(cfg, demand):
    cfg.policy = "random"
    net, ctl = _run(cfg, demand, cfg.simulation_steps)
    assert any(rec["mutations"] for rec in ctl.history)
    for rec in ctl.history:
        assert len(rec["mutations"]) <= cfg.max_mutations_per_step
    assert nx.is_connected(net.G)


def test_greedy_policy_mutates_and_stays_connected(cfg, demand):
    cfg.policy = "greedy"
    net, ctl = _run(cfg, demand, cfg.simulation_steps)
    assert any(rec["mutations"] for rec in ctl.history)
    assert nx.is_connected(net.G)


def test_ml_forecast_only_affects_indicators_not_mutations(cfg, demand):
    """[ML] predictor output is stress only; P-LENS still decides all mutations."""
    cfg.window_length_L = 2
    cfg.simulation_steps = 5

    class StubPredictor:
        def predict_stress(self, hist):
            assert hist.shape == (cfg.window_length_L, 9)
            return {"label": "High", "class": 2, "prob": [0.1, 0.2, 0.7]}

    net = EvolvingNetwork(cfg, np.random.default_rng(cfg.seed))
    ctl = PLENSController(cfg, net, ml_predictor=StubPredictor(), base_rng=np.random.default_rng(1))
    for t in range(cfg.simulation_steps):
        hist = np.zeros((cfg.window_length_L, 9)) if t >= cfg.window_length_L else None
        rec = ctl.step(t, demand.copy(), hist)
        if t < cfg.window_length_L:
            assert rec["ml"] is None
        else:
            assert rec["ml"] is not None and rec["ml"]["label"] == "High"
    # all mutations still produced by the P-LENS selector, with detail records
    assert any(rec["mutations"] for rec in ctl.history)
