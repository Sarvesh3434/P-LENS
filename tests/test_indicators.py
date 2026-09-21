"""[DIAGRAM] Observation indicators D, F, C, R — ranges and behaviour."""
from __future__ import annotations

import numpy as np

from network.traffic import route_and_load, scale_demand
from plens.indicators import compute_indicators


def _loaded_net(net, cfg, demand):
    net.set_demand(scale_demand(demand, net, cfg.traffic_intensity))
    route_and_load(net)
    return net


def test_indicator_ranges(net, cfg, demand):
    _loaded_net(net, cfg, demand)
    ind = compute_indicators(net, net.demand)
    for name in ("D", "F", "C", "R"):
        assert ind[name].shape == (cfg.network_size,)
        assert np.all(ind[name] >= 0.0) and np.all(ind[name] <= 1.0), name


def test_node_failure_raises_resilience(net, cfg, demand):
    _loaded_net(net, cfg, demand)
    base = compute_indicators(net, net.demand)["R"].copy()
    net.nodes[3].failed = True
    after = compute_indicators(net, net.demand)["R"]
    assert after[3] == 1.0
    assert after[3] >= base[3]


def test_drift_zero_on_reference(net, cfg, demand):
    """With no mutations and no failures, D must be 0 (topology = reference)."""
    ind = compute_indicators(net, net.demand)
    assert np.allclose(ind["D"], 0.0)


def test_drift_positive_after_mutation(net, cfg, demand):
    net.add_edge(0, 5)
    ind = compute_indicators(net, net.demand)
    assert ind["D"][0] > 0.0 and ind["D"][5] > 0.0


def test_congestion_increases_with_load(net, cfg, demand):
    light = compute_indicators(_loaded_net(net, cfg, demand * 0.1), net.demand)["C"]
    net2_demand = demand * 5.0
    heavy = compute_indicators(_loaded_net(net, cfg, net2_demand), net.demand)["C"]
    assert float(np.mean(heavy)) >= float(np.mean(light)) - 1e-9


def test_failure_deficiency_uses_usable_topology(net, cfg, demand):
    """A failed edge must be invisible to the monitor (usable graph)."""
    _loaded_net(net, cfg, demand)
    e = net.edges[(0, 1)]
    e.failed = True
    g = net.usable_graph()
    assert not g.has_edge(0, 1)
