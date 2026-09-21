"""[DIAGRAM] PP, PD, B — the pressure/debt/budget dashed group of the image."""
from __future__ import annotations

import numpy as np

from plens.pressure import plasticity_pressure
from plens.debt import update_debt, relief_for_mutation
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
    pd = np.zeros(n)
    pp = np.full(n, 0.5)
    pd1 = update_debt(pd, pp, relief=np.zeros(n), cfg=cfg)
    assert np.allclose(pd1, cfg.debt_accumulation * 0.5)
    pd2 = update_debt(pd1, pp, relief=np.zeros(n), cfg=cfg)
    assert np.allclose(pd2, cfg.debt_decay * pd1 + cfg.debt_accumulation * 0.5)


def test_debt_relief_reduces_and_clips_at_zero(cfg):
    n = 4
    pd = np.full(n, 0.2)
    pp = np.zeros(n)
    big_relief = np.full(n, 1.0)
    pd1 = update_debt(pd, pp, relief=big_relief, cfg=cfg)
    assert np.allclose(pd1, 0.0)


def test_relief_only_for_touched_nodes(cfg):
    r = relief_for_mutation(6, [1, 4], cfg, success=True)
    assert r[1] > 0 and r[4] > 0 and r[0] == 0 and r[2] == 0
    r0 = relief_for_mutation(6, [1, 4], cfg, success=False)
    assert np.allclose(r0, 0.0)


def test_debt_disabled_returns_zero(cfg):
    pd = np.full(4, 0.5)
    out = update_debt(pd, np.zeros(4), np.zeros(4), cfg, enabled=False)
    assert np.allclose(out, 0.0)


def test_budget_spends_and_refills_from_debt(cfg):
    b = 5.0
    spent = 2.0
    mean_pd = 0.4
    b1 = update_budget(b, spent, mean_pd, 0.0, 0.0, cfg)
    expected = min(cfg.mutation_budget_max, b - spent + cfg.budget_refill * mean_pd)
    assert np.isclose(b1, expected)


def test_budget_gain_raises_and_stability_lowers(cfg):
    b0 = 3.0
    up = update_budget(b0, 0.0, 0.0, 0.5, 0.0, cfg)
    down = update_budget(b0, 0.0, 0.0, 0.0, 0.5, cfg)
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
