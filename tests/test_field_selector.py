"""[DIAGRAM] MF field generation + constraint-aware selection."""
from __future__ import annotations

import numpy as np

from network.traffic import route_and_load, scale_demand
from plens.mutation_field import generate_field, score_candidate
from plens.selector import select_mutations


def _pressures(net):
    n = net.n
    import numpy as np

    return np.full(n, 0.5), np.full(n, 0.3)


def test_field_enumerates_ops(net, cfg, demand):
    pp, pd = _pressures(net)
    field = generate_field(net, pp, pd, {}, cfg.mutation_budget_initial, cfg)
    ops = {c["op"] for c in field}
    assert {"ADD", "PRUNE", "REWIRE"} <= ops
    # ring of 12: 54 non-edges for ADD, 12 for PRUNE, REWIRE present too
    assert any(c["op"] == "ADD" for c in field)
    assert any(c["op"] == "REWIRE" for c in field)


def test_field_gate_blocks_when_budget_empty(net, cfg, demand):
    pp, pd = _pressures(net)
    field = generate_field(net, pp, pd, {}, budget=0.0, cfg=cfg)
    add = [c for c in field if c["op"] == "ADD"][0]
    assert add["score"] == 0.0  # gate(B_t) = 0 when B_t < min cost


def test_credit_increases_score(net, cfg, demand):
    pp, pd = _pressures(net)
    key = (0, 5)
    mc_no = {}
    mc_yes = {key: 0.8}
    s_no = score_candidate(net, 0, 5, "ADD", pp, pd, mc_no, 5.0, cfg, use_credit=True)
    s_yes = score_candidate(net, 0, 5, "ADD", pp, pd, mc_yes, 5.0, cfg, use_credit=True)
    assert s_yes > s_no


def test_no_credit_ablation_ignores_mc(net, cfg, demand):
    pp, pd = _pressures(net)
    s1 = score_candidate(net, 0, 5, "ADD", pp, pd, {(0, 5): 0.9}, 5.0, cfg, use_credit=False)
    s2 = score_candidate(net, 0, 5, "ADD", pp, pd, {}, 5.0, cfg, use_credit=False)
    assert np.isclose(s1, s2)


def test_selector_respects_budget(net, cfg, demand):
    pp, pd = _pressures(net)
    field = generate_field(net, pp, pd, {}, 5.0, cfg)
    chosen, rejected = select_mutations(net, field, budget=1.0, cfg=cfg, cooldown={})
    spent = sum(m["budget_cost"] for m in chosen)
    assert spent <= 1.0 + 1e-9
    assert len(chosen) <= cfg.max_mutations_per_step


def test_selector_applies_mutations(net, cfg, demand):
    pp, pd = _pressures(net)
    field = generate_field(net, pp, pd, {}, 8.0, cfg)
    chosen, rejected = select_mutations(net, field, budget=8.0, cfg=cfg, cooldown={})
    assert chosen, "expected at least one accepted mutation at default settings"
    for m in chosen:
        d = m["detail"]
        if d.get("added"):
            assert net.G.has_edge(*d["added"])
        if d.get("removed"):
            assert not net.G.has_edge(*d["removed"])


def test_selector_records_rejections(net, cfg, demand):
    pp, pd = _pressures(net)
    field = generate_field(net, pp, pd, {}, 8.0, cfg)
    chosen, rejected = select_mutations(net, field, budget=0.0, cfg=cfg, cooldown={})
    assert not chosen
    assert len(rejected) == len(field)
    assert all("reason" in r for r in rejected)


def test_selector_validates_connectivity(net, cfg, demand):
    """No accepted mutation may disconnect the usable graph."""
    import networkx as nx

    pp, pd = _pressures(net)
    field = generate_field(net, pp, pd, {}, 8.0, cfg)
    select_mutations(net, field, budget=8.0, cfg=cfg, cooldown={})
    assert nx.is_connected(net.G)


def test_cooldown_blocks_repeat(net, cfg, demand):
    from plens import mutations as muts

    pp, pd = _pressures(net)
    field = generate_field(net, pp, pd, {}, 8.0, cfg)
    chosen, _ = select_mutations(net, field, budget=8.0, cfg=cfg, cooldown={})
    edges_touched = set()
    for m in chosen:
        d = m["detail"]
        for e in (d.get("added"), d.get("removed")):
            if e:
                edges_touched.add((min(e), max(e)))
                cd = {(min(e), max(e)): cfg.cooldown_steps}
                ok, reason = muts.validate(net, {"op": "ADD", "i": e[0], "j": e[1]}, cfg, cd)
                assert reason == "cooldown"
