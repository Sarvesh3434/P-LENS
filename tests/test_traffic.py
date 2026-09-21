"""[SIM] Traffic routing, M/M/1 queueing, loss, demand scaling."""
from __future__ import annotations

import numpy as np

from network.traffic import route_and_load, scale_demand


METRIC_KEYS = (
    "latency",
    "throughput",
    "loss",
    "congestion",
    "connected",
    "avg_path_length",
    "n_edges",
)


def _metrics_keys(net, cfg, demand):
    net.set_demand(scale_demand(demand, net, cfg.traffic_intensity))
    m = route_and_load(net)
    for k in METRIC_KEYS:
        assert k in m, k
    return m


def test_metrics_present_and_sane(net, cfg, demand):
    m = _metrics_keys(net, cfg, demand)
    assert m["latency"] > 0.0  # propagation latency is always > 0
    assert 0.0 <= m["throughput"] <= 1.0
    assert 0.0 <= m["loss"] <= 1.0
    assert 0.0 <= m["congestion"] <= 1.0
    assert m["connected"] == 1.0  # ring is connected, nothing failed


def test_scale_demand_hits_target_utilization(net, cfg, demand):
    target = 0.4
    scaled = scale_demand(demand, net, target)
    usable_bw = sum(
        e.bandwidth
        for e in net.edges.values()
        if not e.failed and not net.nodes[e.u].failed and not net.nodes[e.v].failed
    )
    assert np.isclose(scaled.sum(), target * usable_bw, rtol=1e-6)


def test_scale_demand_ignores_failed_bandwidth(net, cfg, demand):
    scaled_full = scale_demand(demand, net, 0.5)
    e = net.edges[(0, 1)]
    e.failed = True
    scaled_less = scale_demand(demand, net, 0.5)
    assert scaled_less.sum() < scaled_full.sum()  # 11 usable links instead of 12


def test_queue_saturates_and_drops(net, cfg, demand):
    """load >= service rate ⇒ queue_delay = 1 and overflow loss. [SIM]"""
    d = np.zeros((cfg.network_size, cfg.network_size))
    d[0, 1] = 5.0 * cfg.mm1_service_rate  # far above mu on the direct link
    net.set_demand(d)
    m = route_and_load(net)
    assert net.edges[(0, 1)].queue_delay == 1.0
    assert m["loss"] > 0.0


def test_latency_is_propagation_plus_queue(net, cfg, demand):
    """At tiny load the latency ≈ hops × prop_latency (queueing ≈ 0). [SIM]"""
    d = np.zeros((cfg.network_size, cfg.network_size))
    d[0, 1] = 0.01 * cfg.mm1_service_rate
    net.set_demand(d)
    m = route_and_load(net)
    assert 0.0 < m["latency"] < 10 * cfg.default_prop_latency  # ≤ ~10 hops, no queueing


def test_failed_edge_reroutes_traffic(net, cfg, demand):
    """Failing one ring edge reroutes traffic onto the remaining ring edges. [SIM]

    The rerouted flow must land on the ring edges *adjacent* to the failed one,
    not on (0,11): before the failure (0,11) is the shortcut side of the ring
    carrying many long-haul pairs, after it the ring degenerates to a path and
    (0,11) becomes a pendant edge carrying only node 0's traffic. [ASSUMPTION]
    """
    net.set_demand(scale_demand(demand, net, cfg.traffic_intensity))
    route_and_load(net)
    load_before = net.edges[(10, 11)].load
    net.edges[(0, 1)].failed = True
    m = route_and_load(net)
    assert net.edges[(0, 1)].load == 0.0
    # traffic that used (0,1) is pushed around the ring through (10,11)
    assert net.edges[(10, 11)].load > load_before
    assert net.edges[(1, 2)].load > 0.0  # the far side of the break still carries flow
    assert m["connected"] == 1.0  # ring minus one edge is still connected


def test_isolated_node_drops_demand(net, cfg, demand):
    """Two failed edges isolate node 0 on the ring; its demand must be lost. [SIM]"""
    net.edges[(0, 1)].failed = True
    net.edges[(0, 11)].failed = True
    net.set_demand(scale_demand(demand, net, cfg.traffic_intensity))
    m = route_and_load(net)
    assert m["loss"] > 0.0
    assert net.nodes[0].loss == 1.0


def test_congestion_metric_reflects_load(net, cfg, demand):
    d_light = scale_demand(demand, net, 0.05)
    net.set_demand(d_light)
    m_light = route_and_load(net)
    d_heavy = scale_demand(demand, net, 1.5)
    net.set_demand(d_heavy)
    m_heavy = route_and_load(net)
    assert m_heavy["congestion"] > m_light["congestion"]
    assert m_heavy["latency"] > m_light["latency"]
    assert m_heavy["throughput"] < m_light["throughput"]
