"""[DIAGRAM] ADD / PRUNE / REWIRE — independently unit-testable. Prompt §8."""
from __future__ import annotations

import networkx as nx

from network.graph import canon


def test_add_valid(net, cfg):
    from plens import mutations as muts

    i, j = 0, 5  # 0-5 not an edge of the ring
    cand = {"op": "ADD", "i": i, "j": j}
    ok, reason = muts.validate(net, cand, cfg, {})
    assert ok, reason
    applied, detail = muts.apply(net, cand)
    assert applied
    assert net.G.has_edge(i, j)
    assert detail["added"] == [i, j]
    # ADD must keep the graph connected (ring + chord is connected)
    assert nx.is_connected(net.G)


def test_add_rejects_existing_and_self_loop(net, cfg):
    from plens import mutations as muts

    ok, reason = muts.validate(net, {"op": "ADD", "i": 0, "j": 1}, cfg, {})
    assert not ok and reason == "already_exists"
    ok, reason = muts.validate(net, {"op": "ADD", "i": 0, "j": 0}, cfg, {})
    assert not ok and reason == "self_loop"


def test_add_rejects_degree_cap(net, cfg):
    from plens import mutations as muts

    cfg.max_degree = 2  # ring nodes already have degree 2
    ok, reason = muts.validate(net, {"op": "ADD", "i": 0, "j": 5}, cfg, {})
    assert not ok and reason == "degree_cap"


def test_add_rejects_failed_endpoint(net, cfg):
    from plens import mutations as muts

    net.nodes[5].failed = True
    ok, reason = muts.validate(net, {"op": "ADD", "i": 0, "j": 5}, cfg, {})
    assert not ok and reason == "endpoint_failed"


def test_prune_valid_keeps_connectivity(net, cfg):
    from plens import mutations as muts

    net.add_edge(0, 5)  # redundant chord
    cand = {"op": "PRUNE", "i": 0, "j": 5}
    ok, reason = muts.validate(net, cand, cfg, {})
    assert ok, reason
    applied, detail = muts.apply(net, cand)
    assert applied
    assert not net.G.has_edge(0, 5)
    assert nx.is_connected(net.G)
    assert detail["removed"] == [0, 5]


def test_prune_rejects_disconnect(net, cfg):
    from plens import mutations as muts

    ok, reason = muts.validate(net, {"op": "PRUNE", "i": 0, "j": 1}, cfg, {})
    assert not ok and reason == "would_disconnect"


def test_prune_rejects_missing_edge(net, cfg):
    from plens import mutations as muts

    ok, reason = muts.validate(net, {"op": "PRUNE", "i": 0, "j": 5}, cfg, {})
    assert not ok and reason == "missing_edge"


def test_rewire_moves_one_endpoint_and_keeps_connectivity(net, cfg):
    from plens import mutations as muts

    # ring ... 8-9-10-11-0 ... rewire (0,11) -> (0,5)
    # node 11 must keep min_degree after losing (0,11): give it a real spare chord
    # (10,11) is already a ring edge, so chord (9,11) provides the headroom
    net.add_edge(9, 11)
    cand = {"op": "REWIRE", "i": 0, "j": 11, "l": 5}
    ok, reason = muts.validate(net, cand, cfg, {})
    assert ok, reason
    applied, detail = muts.apply(net, cand)
    assert applied
    assert not net.G.has_edge(0, 11)
    assert net.G.has_edge(0, 5)
    assert nx.is_connected(net.G)
    assert detail["added"] == [0, 5] and detail["removed"] == [0, 11]


def test_rewire_rejects_min_degree_violation(net, cfg):
    from plens import mutations as muts

    # node 11 would drop to degree 1 (< min_degree=2) if (0,11) is removed
    ok, reason = muts.validate(net, {"op": "REWIRE", "i": 0, "j": 11, "l": 5}, cfg, {})
    assert not ok and reason == "min_degree"


def test_rewire_rejects_bad_target(net, cfg):
    from plens import mutations as muts

    ok, reason = muts.validate(net, {"op": "REWIRE", "i": 0, "j": 11, "l": 1}, cfg, {})
    assert not ok and reason == "already_exists"


def test_revert_one_restores(net, cfg):
    from plens import mutations as muts

    before = sorted(map(list, net.G.edges()))
    cand = {"op": "ADD", "i": 0, "j": 5}
    applied, detail = muts.apply(net, cand)
    assert applied
    muts.revert_one(net, detail)
    assert sorted(map(list, net.G.edges())) == before


def test_cooldown_blocks(net, cfg):
    from plens import mutations as muts

    cd = {canon(0, 5): 2}
    ok, reason = muts.validate(net, {"op": "ADD", "i": 0, "j": 5}, cfg, cd)
    assert not ok and reason == "cooldown"


def test_reverse_of_detail():
    from plens import mutations as muts

    detail = {"op": "ADD", "added": [0, 5], "removed": None}
    rev = muts.reverse_of(detail)
    assert rev["added"] is None and rev["removed"] == [0, 5]
