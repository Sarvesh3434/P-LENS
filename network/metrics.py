"""[SIM] Network-level snapshot metrics."""

from __future__ import annotations

from typing import Dict

import networkx as nx

from network.graph import EvolvingNetwork


def snapshot(net: EvolvingNetwork) -> Dict[str, float]:
    m = dict(net.last_metrics)
    m["n_nodes_alive"] = float(len(net.alive_nodes()))
    m["n_edges"] = float(net.G.number_of_edges())
    try:
        m["edge_connectivity"] = float(nx.edge_connectivity(net.G))
    except Exception:
        m["edge_connectivity"] = 0.0
    return m
