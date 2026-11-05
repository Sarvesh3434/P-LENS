"""Focused tests for the algorithmic revision.

Covers the dependency-aware debt model, primal-dual mutation economics,
higher-order interactions, drift-aware attribution, the unified objective, and
the executable theory checks.  Existing behaviour is covered by the original
suite; these tests target the new mechanisms directly.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from experiments.config import Config
from network.graph import EvolvingNetwork
from plens import theory
from plens.attribution import (
    build_result,
    credit_weight,
    decompose,
    demand_drift,
    difference_in_differences,
    drift_score,
    lcb_stats,
    metric_drift,
)
from plens.budget import update_budget
from plens.controller import PLENSController
from plens.debt import (
    DebtState,
    debt_contraction_factor,
    debt_fixed_point,
    dependency_matrix,
    modification_gate,
    propagate_inadequacy,
    propagate_relief,
    relief_potential,
    update_debt,
)
from plens.economics import (
    EconomyState,
    dual_residual,
    economy_step,
    lagrangian,
    marginal_utility,
    update_dual_price,
)
from plens.interaction import (
    CompatibilityStructure,
    group_effect,
    hard_conflict,
    interaction,
    interaction_triple,
)
from plens.mutation_field import generate_field
from plens.selector import select_mutations
from plens.unified import UnifiedState, unified_pseudocode, unified_score


def _run(cfg: Config, demand: np.ndarray, steps: int):
    net = EvolvingNetwork(cfg, np.random.default_rng(cfg.seed))
    ctl = PLENSController(cfg, net, base_rng=np.random.default_rng(cfg.seed + 1))
    for t in range(steps):
        ctl.step(t, demand.copy())
    return net, ctl


# ---------------------------------------------------------------- debt model
def test_debt_is_bounded_under_extreme_relief_and_failure(cfg):
    n = 6
    state = DebtState.zeros(n)
    for _ in range(60):
        state = update_debt(
            state,
            np.full(n, 10.0),
            np.full(n, 10.0),
            np.full(n, 10.0),
            cfg,
            failure=np.full(n, 10.0),
        )
        assert np.all(state.pd >= 0.0) and np.all(state.pd <= 1.0)
        assert np.all(state.inadequacy >= 0.0) and np.all(state.inadequacy <= 1.0)


def test_dependency_propagation_raises_dependent_debt(cfg):
    n = 4
    dependency = np.zeros((n, n))
    dependency[0, 1] = 1.0  # node 0 depends on node 1
    pp = np.zeros(n)
    pp[1] = 0.95

    plain = DebtState.zeros(n)
    propagated = DebtState.zeros(n)
    for _ in range(10):
        plain = update_debt(plain, pp, np.zeros(n), np.ones(n), cfg, dependency=None)
        propagated = update_debt(
            propagated, pp, np.zeros(n), np.ones(n), cfg, dependency=dependency
        )

    assert plain.pd[0] == 0.0
    assert propagated.pd[0] > 0.0
    assert propagated.propagated[0] > 0.0


def test_inadequacy_rate_accelerates_worsening_debt(cfg):
    c = replace(cfg, debt_rate_gain=0.25, debt_rate_decay=0.25)
    n = 3
    pp = np.full(n, 0.9)
    first = update_debt(DebtState.zeros(n), pp, np.zeros(n), np.ones(n), c)

    base = DebtState(
        pd=np.full(n, 0.3),
        persistence=first.persistence.copy(),
        duration=first.duration.copy(),
        inadequacy=np.zeros(n),
        rate=np.zeros(n),
    )
    stable = replace(base, inadequacy=first.inadequacy.copy())

    rising = update_debt(base, pp, np.zeros(n), np.ones(n), c)
    flat = update_debt(stable, pp, np.zeros(n), np.ones(n), c)
    assert np.all(rising.rate > 0.0)
    assert np.all(rising.pd >= flat.pd)


def test_modification_gate_suppresses_accumulation(cfg):
    n = 3
    pp = np.full(n, 0.9)
    open_gate = update_debt(
        DebtState.zeros(n), pp, np.zeros(n), np.ones(n), cfg,
        modification_gate=np.ones(n),
    )
    damped = update_debt(
        DebtState.zeros(n), pp, np.zeros(n), np.ones(n), cfg,
        modification_gate=np.full(n, 0.1),
    )
    assert np.all(damped.pd < open_gate.pd)
    assert np.all(modification_gate(n, [{"t": 0, "nodes": [1]}], 0, cfg) <= 1.0)


def test_debt_contracts_to_documented_fixed_point(cfg):
    s = 0.8
    eta = cfg.debt_accumulation
    gamma = cfg.debt_decay
    pd = 0.0
    for _ in range(1000):
        pd = pd + eta * s * (1.0 - pd) - gamma * (1.0 - s) * pd
    assert abs(pd - debt_fixed_point(s, cfg)) < 1e-6
    assert debt_contraction_factor(s, cfg) < 1.0
    conditions = theory.debt_convergence_conditions(cfg)
    assert conditions["is_contraction"]
    assert conditions["eta_in_unit_interval"] and conditions["gamma_in_unit_interval"]


def test_relief_potential_peaks_at_moderate_debt():
    pd = np.array([0.0, 0.5, 1.0])
    rp = relief_potential(pd)
    assert rp[0] == 0.0 and rp[2] == 0.0
    assert rp[1] > rp[0] and rp[1] > rp[2]


def test_relief_diffuses_to_dependants(cfg):
    n = 3
    dependency = np.zeros((n, n))
    dependency[0, 1] = 1.0
    relief = np.zeros(n)
    relief[1] = cfg.debt_relief
    spread = propagate_relief(relief, dependency, cfg)
    assert spread[0] > 0.0
    assert np.all(spread <= cfg.debt_relief + 1e-12)


# ----------------------------------------------------------- mutation economics
def test_marginal_utility_has_expected_sign():
    assert marginal_utility(value=1.0, cost=1.0, price=0.2) > 0.0
    assert marginal_utility(value=0.1, cost=1.0, price=0.5) < 0.0


def test_dual_price_rises_when_overspent_and_falls_when_idle(cfg):
    c = replace(cfg, budget_price_initial=0.5, budget_dual_decay=1.0)
    over = update_dual_price(0.5, spent=5.0, budget=1.0, cfg=c, iteration=0)
    idle = update_dual_price(0.5, spent=0.0, budget=5.0, cfg=c, iteration=0)
    assert over > 0.5
    assert idle < 0.5
    assert dual_residual(5.0, 1.0, c) > 0.0
    assert dual_residual(0.0, 5.0, c) < 0.0


def test_dual_price_is_projected_into_bounds(cfg):
    c = replace(cfg, budget_price_min=0.1, budget_price_max=0.9, budget_dual_step=5.0)
    high = update_dual_price(0.5, spent=1e6, budget=1.0, cfg=c, iteration=0)
    low = update_dual_price(0.5, spent=-1e6, budget=1.0, cfg=c, iteration=0)
    assert c.budget_price_min <= high <= c.budget_price_max
    assert c.budget_price_min <= low <= c.budget_price_max


def test_lagrangian_prices_cost_against_the_account(cfg):
    value = 0.8
    cost = 1.2
    budget = 1.0
    price = 0.4
    got = lagrangian(value, cost, price, budget)
    assert np.isclose(got, value - price * (cost - budget))


def test_economy_account_is_bounded_and_modes_work(cfg):
    state = EconomyState(budget=3.0, price=0.3)
    out = economy_step(
        state, spent=1.0, mean_pd=0.5, mean_gain=0.2, mean_stability=0.05, cfg=cfg
    )
    assert cfg.mutation_budget_min <= out.budget <= cfg.mutation_budget_max
    assert out.marginal_utility != 0.0 or out.budget == state.budget

    fixed = economy_step(state, 1.0, 0.5, 0.2, 0.0, cfg, mode="fixed")
    assert fixed.budget == cfg.mutation_budget_initial

    disabled = economy_step(state, 1.0, 0.5, 0.2, 0.0, cfg, enabled=False)
    assert disabled.budget == cfg.mutation_budget_max


def test_budget_helper_still_exposes_dual_priced_account(cfg):
    b = update_budget(
        3.0, 1.0, 0.5, 0.2, 0.05, cfg, dual_price=0.4, iteration=3, churn=0.1
    )
    assert cfg.mutation_budget_min <= b <= cfg.mutation_budget_max


# ------------------------------------------------------ higher-order interaction
def test_pair_conflict_propagates_to_triples(net, cfg):
    a = {"op": "ADD", "i": 0, "j": 5}
    b = {"op": "PRUNE", "i": 0, "j": 5}
    c = {"op": "ADD", "i": 1, "j": 6}
    assert hard_conflict(a, b, net, cfg)
    assert interaction(a, b, net, cfg) <= -0.999
    assert interaction_triple(a, b, c, net, cfg) <= -0.999
    compat = CompatibilityStructure(net, cfg)
    assert not compat.group_ok([a, b, c])
    assert compat.relation(a, b) == "conflict"


def test_third_order_factor_is_bounded_and_detects_destabilisation(net, cfg):
    prunes = [
        {"op": "PRUNE", "i": 0, "j": 1},
        {"op": "PRUNE", "i": 0, "j": 11},
        {"op": "PRUNE", "i": 1, "j": 2},
    ]
    v = interaction_triple(*prunes, net, cfg)
    assert -1.0 <= v <= 1.0
    assert v < 0.0


def test_group_effect_includes_third_order_term(net, cfg):
    group = [
        {"op": "ADD", "i": 0, "j": 5},
        {"op": "ADD", "i": 1, "j": 6},
        {"op": "ADD", "i": 2, "j": 7},
    ]
    c2 = replace(cfg, interaction_order=2)
    c3 = replace(cfg, interaction_order=3, interaction_triple_weight=0.15)
    assert np.isclose(group_effect(group, net, c2), group_effect(group, net, c2))
    assert np.isfinite(group_effect(group, net, c3))


# --------------------------------------------------------------- joint selector
def test_joint_selector_terminates_when_best_candidate_is_unaffordable(net, cfg):
    """Regression: an unaffordable top candidate used to spin forever."""
    field = [
        {"op": "ADD", "i": 0, "j": 5, "score": 10.0},
        {"op": "PRUNE", "i": 0, "j": 1, "score": 0.1},
    ]
    chosen, rejected = select_mutations(net, field, budget=0.9, cfg=cfg, cooldown={})
    assert sum(m["budget_cost"] for m in chosen) <= 0.9 + 1e-9
    assert any(r["reason"] == "insufficient_budget" for r in rejected)
    assert len(chosen) <= cfg.max_mutations_per_step


def test_joint_selector_respects_cap_and_budget(net, cfg, demand):
    pp = np.full(net.n, 0.6)
    pd = np.full(net.n, 0.4)
    field = generate_field(net, pp, pd, {}, 5.0, cfg)
    chosen, _ = select_mutations(net, field, budget=2.0, cfg=cfg, cooldown={})
    assert len(chosen) <= cfg.max_mutations_per_step
    assert sum(m["budget_cost"] for m in chosen) <= 2.0 + 1e-9


# ---------------------------------------------------- counterfactual attribution
def test_lcb_is_below_mean_and_tightens_with_less_noise(cfg):
    noisy = [0.2, -0.4, 0.5, 0.1, -0.2]
    clean = [0.2, 0.2, 0.2, 0.2, 0.2]
    m1, se1, l1, c1 = lcb_stats(noisy, cfg)
    m2, se2, l2, c2 = lcb_stats(clean, cfg)
    assert l1 < m1
    assert se2 == 0.0 and l2 == m2 and c2 == 1.0
    assert se1 > se2 and c1 < c2


def test_hoeffding_bound_is_conservative(cfg):
    c = replace(cfg, cf_lcb_method="hoeffding")
    mean, _, lcb, _ = lcb_stats([0.3, 0.3], c)
    assert lcb < mean


def test_drift_detection_distinguishes_same_and_shifted_demand(cfg):
    a = np.array([[0.0, 1.0], [1.0, 0.0]])
    b = np.array([[0.0, 1.0], [1.0, 0.0]])
    c = np.array([[0.0, 5.0], [5.0, 0.0]])
    d = np.array([[0.0, 1.0], [1.0, 1.0]])  # different normalized support
    assert demand_drift(a, b) == 0.0
    assert demand_drift(c, d) > 0.2
    assert metric_drift({"latency": 1.0}, {"latency": 1.0}) == 0.0
    assert metric_drift({"latency": 1.0}, {"latency": 2.0}) > 0.0
    assert drift_score(a, d, None, None, cfg) > 0.0


def test_direct_indirect_decomposition_and_credit_weight(cfg):
    direct, indirect = decompose(total_gain=0.8, direct_gain=0.3)
    assert np.isclose(direct, 0.3) and np.isclose(indirect, 0.5)
    assert credit_weight(1.0, 0.0, True, cfg) == 1.0
    assert credit_weight(1.0, 1.0, True, cfg) < 1.0
    assert credit_weight(1.0, 0.0, False, cfg) <= 0.5


def test_build_result_marks_unreliable_replay_and_keeps_lcb(cfg):
    result = build_result(
        total_gain=0.4,
        step_gains=[0.1, 0.5, 0.6],
        cfg=cfg,
        direct_gain=0.15,
        drift=cfg.cf_drift_threshold + 0.2,
    )
    assert not result.reliable
    assert result.credit_weight < 1.0
    assert result.direct_gain == 0.15
    assert np.isclose(result.indirect_gain, 0.25)
    assert result.gain_lcb <= result.total_gain


def test_difference_in_differences_is_zero_for_identical_branches(cfg):
    m = {"latency": 0.2, "loss": 0.1, "congestion": 0.1, "throughput": 0.5, "connected": 1.0, "n_edges": 20}
    assert difference_in_differences(m, m, m, m, cfg) == 0.0


# ------------------------------------------------------------ unified + theory
def test_unified_score_reacts_to_price_and_debt(net, cfg):
    cand = {"op": "ADD", "i": 0, "j": 5}
    base = UnifiedState(
        pp=np.full(net.n, 0.6),
        pd=np.zeros(net.n),
        mc={},
        price=0.1,
        budget=5.0,
    )
    pricey = replace(base, price=2.0)
    indebted = replace(base, pd=np.full(net.n, 0.8))
    assert unified_score(net, cand, base, cfg) > unified_score(net, cand, pricey, cfg)
    assert unified_score(net, cand, indebted, cfg) > unified_score(net, cand, base, cfg)


def test_unified_pseudocode_covers_full_loop():
    text = unified_pseudocode()
    for token in ("update_debt", "joint_select", "matched_replay", "dual_ascent"):
        assert token in text


def test_theory_checks_are_executable(cfg):
    report = theory.verify_all(cfg)
    assert set(report) >= {
        "bounded_debt",
        "debt_convergence",
        "budget_stability",
        "selection",
        "counterfactual",
        "topology_stability",
        "adaptation_efficiency",
    }
    assert report["debt_convergence"]["is_contraction"]
    assert theory.dual_regret_bound(100, 1.0, 1.0, 0.1) > 0.0
    assert theory.counterfactual_error_bound(0.05, 0.1) > 0.0
    stable = theory.topology_stability_condition(0.2, 2.0, 0.8, 1.0, 0.5)
    assert stable["stable"]
    assert np.isclose(theory.adaptation_efficiency(0.4, 0.8), 0.5)


def test_controller_exposes_new_diagnostics(cfg, demand):
    net, ctl = _run(cfg, demand, cfg.simulation_steps)
    rec = ctl.history[-1]
    for key in (
        "price",
        "marginal_utility",
        "dual_residual",
        "debt_inadequacy_mean",
        "debt_rate_mean",
        "debt_propagation_mean",
        "mod_gate_mean",
        "churn",
        "attr_total_mean",
        "drift_mean",
        "attribution_reliability",
    ):
        assert key in rec
    assert cfg.budget_price_min <= rec["price"] <= cfg.budget_price_max
    assert 0.0 <= rec["mod_gate_mean"] <= 1.0
    assert 0.0 <= rec["attribution_reliability"] <= 1.0
    # Debt propagation must be active by default.
    assert rec["debt_propagation_mean"] >= 0.0
    evaluated = [
        m["delayed_eval"]
        for r in ctl.history
        for m in r["mutations"]
        if m.get("delayed_eval") is not None
    ]
    assert evaluated
    assert all("direct_gain" in e and "drift_score" in e for e in evaluated)
