"""[SIM] Failures and recovery of nodes and edges.

Model [ASSUMPTION]:
- Each alive node fails independently per step with probability failure_rate_node.
- Each non-failed edge whose endpoints are alive fails independently with
  probability failure_rate_edge.
- A failed element repairs after `repair_steps` steps. A node failure also takes
  its incident edges down; those edges come back with the node, not on their own
  repair timer (they carry `failed_by_node = True`).
- While an edge is down it is NOT removed from the logical graph self.G. Routing
  and the monitor only ever see `usable_graph()`, which excludes failed elements.
  This preserves per-edge state (history, bandwidth) across failures.

[DIAGRAM] this implements the "Failures" bullet of the Dynamic Network Environment.
"""

from __future__ import annotations

import numpy as np

from network.graph import EvolvingNetwork, canon


def _edge_down(net: EvolvingNetwork, e) -> None:
    """Mark an edge failed without touching the logical graph. [SIM]"""
    e.failed = True
    e.failed_by_node = False


def apply_failures(net: EvolvingNetwork) -> None:
    cfg = net.cfg
    rng = net.rng

    # --- recovery (edges) ---
    for e in net.edges.values():
        if e.failed and not e.failed_by_node:
            e.repair_in -= 1
            if e.repair_in <= 0:
                e.failed = False
                e.failed_by_node = False

    # --- recovery (nodes): a node comes back and re-activates its incident edges ---
    for nd in net.nodes.values():
        if nd.failed:
            nd.repair_in -= 1
            if nd.repair_in <= 0:
                nd.failed = False
                for u, v in list(net.G.edges(nd.node_id)):
                    e = net.edges.get(canon(u, v))
                    if e is not None and e.failed_by_node:
                        e.failed = False
                        e.failed_by_node = False

    mode = cfg.failure_mode
    if mode == "random":
        # --- new node failures: incident edges go down with the node ---
        for nd in net.nodes.values():
            if not nd.failed and rng.random() < cfg.failure_rate_node:
                _fail_node(net, nd)
        for e in list(net.edges.values()):
            if not e.failed:
                nd_u = net.nodes[e.u]
                nd_v = net.nodes[e.v]
                if not (nd_u.failed or nd_v.failed) and rng.random() < cfg.failure_rate_edge:
                    _fail_edge(net, e)
        return

    if mode == "recovery":
        # Outage/recovery storm: a burst every ten steps, few isolated failures.
        burst = net.step % 10 == 0
        n_node = 2 if burst else 0
        n_edge = 3 if burst else 0
        if not burst and rng.random() < cfg.failure_rate_node:
            n_node = 1
    elif mode == "multi":
        n_node = max(1, int(round(cfg.network_size * cfg.failure_rate_node * 4.0)))
        n_edge = max(2, int(round(cfg.network_size * cfg.failure_rate_edge * 3.0)))
    else:  # targeted
        n_node = 1 if rng.random() < min(0.95, cfg.failure_rate_node * 12.0) else 0
        n_edge = 1 if rng.random() < min(0.95, cfg.failure_rate_edge * 8.0) else 0

    _targeted_nodes(net, n_node)
    _targeted_edges(net, n_edge)


def _fail_node(net: EvolvingNetwork, nd) -> None:
    nd.failed = True
    nd.repair_in = net.cfg.repair_steps
    for u, v in list(net.G.edges(nd.node_id)):
        e = net.edges.get(canon(u, v))
        if e is not None and not e.failed:
            e.failed = True
            e.failed_by_node = True


def _fail_edge(net: EvolvingNetwork, e) -> None:
    _edge_down(net, e)
    e.repair_in = net.cfg.repair_steps


def _targeted_nodes(net: EvolvingNetwork, count: int) -> None:
    """Fail high-degree / high-utilization nodes first (targeted adversary). [SIM]"""
    if count <= 0:
        return
    alive = [i for i, nd in net.nodes.items() if not nd.failed]
    if not alive:
        return
    w = np.array(
        [1.0 + net.degree(i) + 2.0 * net.nodes[i].utilization for i in alive], dtype=float
    )
    w = w / w.sum()
    k = min(count, len(alive))
    for idx in net.rng.choice(len(alive), size=k, replace=False, p=w):
        _fail_node(net, net.nodes[int(alive[int(idx)])])


def _targeted_edges(net: EvolvingNetwork, count: int) -> None:
    """Fail the most loaded usable edges first (targeted adversary). [SIM]"""
    if count <= 0:
        return
    usable = [
        e
        for e in net.edges.values()
        if not e.failed and not net.nodes[e.u].failed and not net.nodes[e.v].failed
    ]
    if not usable:
        return
    w = np.array([1.0 + 4.0 * e.utilization + 2.0 * e.load / max(e.bandwidth, 1.0) for e in usable])
    w = w / w.sum()
    k = min(count, len(usable))
    for idx in net.rng.choice(len(usable), size=k, replace=False, p=w):
        _fail_edge(net, usable[int(idx)])

