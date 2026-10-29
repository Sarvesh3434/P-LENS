"""[DIAGRAM] PP, PD, B — the pressure/debt/budget dashed group of the image."""
from __future__ import annotations

import numpy as np

from plens.pressure import plasticity_pressure
from plens.debt import DebtState, update_debt, relief_for_mutation
from plens.budget import update_budget, gate


def _ind(n, value):
    return {k: np.full(n, value) for k in ("D", "F", "C", "R")}


def test_pressure_uniform_inputs(cfg):
    n = 6
    pp = plasticity_pressure(_ind(n, 0.5), cfg)
    assert np.allclose(pp, 0.5)


def test_pressure_matches_weighted_mean(cfg):
    n = 4
    ind = {"D": np.zeros(n), "F": np.ones(n), "C": np.zeros(n), "R": np.zeros(n)}
    pp = plasticity_pressure(ind, cfg, prev=None)
    w = cfg.pressure_weights
    expected = w["F"] / (w["D"] + w["F"] + w["C"] + w["R"])
    assert np.allclose(pp, expected)


def test_pressure_ema_blends_with_prev(cfg):
    n = 4
    ind = _ind(n, 1.0)
    prev = np.full(n, 0.0)
    pp = plasticity_pressure(ind, cfg, prev=prev)
    ema = cfg.pressure_ema
    assert np.allclose(pp, ema * 1.0 + (1 - ema) * 0.0)


def test_pressure_clipped(cfg):
    pp = plasticity_pressure(_ind(4, 5.0), cfg)
    assert np.all(pp <= 1.0) and np.all(pp >= 0.0)


def test_debt_accumulates_and_decays(cfg):
    n = 4
    state = DebtState.zeros(n)
    pp = np.full(n, 0.8)
    crit = np.ones(n)
    s1 = update_debt(state, pp, np.zeros(n), crit, cfg)
    assert np.all(s1.pd > 0) and np.all(s1.persistence > 0)
    s2 = update_debt(s1, pp, np.zeros(n), crit, cfg)
    assert np.all(s2.pd >= s1.pd)
    s3 = update_debt(s2, np.zeros(n), np.zeros(n), crit, cfg)
    assert np.all(s3.pd < s2.pd)


def test_debt_relief_reduces_and_clips_at_zero(cfg):
    n = 4
    state = DebtState(pd=np.full(n, 0.2), persistence=np.zeros(n), duration=np.zeros(n))
    pp = np.zeros(n)
    big_relief = np.full(n, 1.0)
    s1 = update_debt(state, pp, big_relief, np.ones(n), cfg)
    assert np.allclose(s1.pd, 0.0)


def test_relief_only_for_touched_nodes(cfg):
    r = relief_for_mutation(6, [1, 4], cfg, success=True)
    assert r[1] > 0 and r[4] > 0 and r[0] == 0 and r[2] == 0
    r0 = relief_for_mutation(6, [1, 4], cfg, success=False)
    assert np.allclose(r0, 0.0)


def test_debt_disabled_returns_zero(cfg):
    state = DebtState(pd=np.full(4, 0.5), persistence=np.zeros(4), duration=np.zeros(4))
    out = update_debt(state, np.zeros(4), np.zeros(4), np.ones(4), cfg, enabled=False)
    assert np.allclose(out.pd, 0.0)


def test_budget_spends_and_refills_from_debt(cfg):
    b = 5.0
    spent = 2.0
    mean_pd = 0.4
    b1 = update_budget(b, spent, mean_pd, 0.0, 0.0, cfg, mode="legacy")
    expected = min(cfg.mutation_budget_max, b - spent + cfg.budget_refill * mean_pd)
    assert np.isclose(b1, expected)


def test_budget_gain_raises_and_stability_lowers(cfg):
    from dataclasses import replace

    c = replace(cfg, budget_base_refill=0.0)
    b0 = 3.0
    up = update_budget(b0, 0.0, 0.0, 0.5, 0.0, c)
    down = update_budget(b0, 0.0, 0.0, 0.0, 0.5, c)
    assert up > b0
    assert down < b0


def test_budget_clamped_to_bounds(cfg):
    huge = update_budget(cfg.mutation_budget_max, -100.0, 0.0, 0.0, 0.0, cfg)
    assert huge == cfg.mutation_budget_max
    tiny = update_budget(cfg.mutation_budget_min, 100.0, 0.0, 0.0, 0.0, cfg)
    assert tiny == cfg.mutation_budget_min


def test_budget_disabled_pins_to_max(cfg):
    out = update_budget(0.0, 0.0, 0.0, 0.0, 0.0, cfg, enabled=False)
    assert out == cfg.mutation_budget_max


def test_gate_requires_min_cost(cfg):
    assert gate(0.0, cfg.mutation_cost["ADD"]) == 0.0
    assert gate(cfg.mutation_cost["ADD"], cfg.mutation_cost["ADD"]) == 1.0

