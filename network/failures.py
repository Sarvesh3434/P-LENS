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

    # --- new node failures: incident edges go down with the node ---
    for nd in net.nodes.values():
        if not nd.failed and rng.random() < cfg.failure_rate_node:
            nd.failed = True
            nd.repair_in = cfg.repair_steps
            for u, v in list(net.G.edges(nd.node_id)):
                e = net.edges.get(canon(u, v))
                if e is not None and not e.failed:
                    e.failed = True
                    e.failed_by_node = True

    # --- new edge failures (only edges whose endpoints are alive) ---
    for e in list(net.edges.values()):
        if not e.failed:
            nd_u = net.nodes[e.u]
            nd_v = net.nodes[e.v]
            if not (nd_u.failed or nd_v.failed) and rng.random() < cfg.failure_rate_edge:
                _edge_down(net, e)
                e.repair_in = cfg.repair_steps
