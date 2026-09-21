"""[SIM] Evolving undirected graph with node/edge state. Topology is NOT taken from the dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from experiments.config import Config

Edge = Tuple[int, int]


def canon(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


@dataclass
class NodeState:
    node_id: int
    capacity: float
    failed: bool = False
    demand: float = 0.0
    provisioned: float = 0.0
    utilization: float = 0.0
    queue_delay: float = 0.0
    loss: float = 0.0
    traffic_in: float = 0.0
    traffic_out: float = 0.0
    repair_in: int = 0


@dataclass
class EdgeState:
    u: int
    v: int
    bandwidth: float
    prop_latency: float
    reliability: float = 0.99
    failed: bool = False
    load: float = 0.0
    utilization: float = 0.0
    packet_load: float = 0.0
    repair_in: int = 0
    # [SIM] True while the edge is down only because an endpoint node is down.
    # Such edges recover with the node, not on their own repair timer. [ASSUMPTION]
    failed_by_node: bool = False
    queue_delay: float = 0.0  # normalized M/M/1 queueing term in [0, 1] [ASSUMPTION]
    mutation_history: List[dict] = field(default_factory=list)


class EvolvingNetwork:
    """G_t = (V, E_t). Mutations change edges; node set is fixed. [DIAGRAM] Evolving Network Topology."""

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.n = cfg.network_size
        self.G = self._build_initial(cfg)
        self.reference = self.G.copy()
        self.nodes: Dict[int, NodeState] = {
            i: NodeState(i, cfg.node_capacity) for i in range(self.n)
        }
        self.edges: Dict[Edge, EdgeState] = {}
        self._sync_edges_from_graph()
        self.demand = np.zeros((self.n, self.n), dtype=float)
        self.step = 0
        self.last_metrics: Dict[str, float] = {}

    def _build_initial(self, cfg: Config) -> nx.Graph:
        kind = cfg.initial_topology
        n = cfg.network_size
        if kind == "ring":
            g = nx.cycle_graph(n)
        elif kind == "erdos_renyi":
            g = nx.erdos_renyi_graph(n, cfg.er_p, seed=int(self.rng.integers(0, 1_000_000)))
            while not nx.is_connected(g):
                g = nx.erdos_renyi_graph(n, min(0.99, cfg.er_p + 0.05), seed=int(self.rng.integers(0, 1_000_000)))
        elif kind == "barabasi_albert":
            g = nx.barabasi_albert_graph(n, cfg.ba_m, seed=int(self.rng.integers(0, 1_000_000)))
        else:
            g = nx.watts_strogatz_graph(n, cfg.sw_k, cfg.sw_p, seed=int(self.rng.integers(0, 1_000_000)))
        if not nx.is_connected(g):
            # [SIM] force connectivity by linking components
            comps = list(nx.connected_components(g))
            for a, b in zip(comps, comps[1:]):
                g.add_edge(next(iter(a)), next(iter(b)))
        return g

    def _sync_edges_from_graph(self) -> None:
        live = {canon(u, v) for u, v in self.G.edges()}
        for e in list(self.edges):
            if e not in live:
                del self.edges[e]
        for u, v in live:
            if (u, v) not in self.edges:
                self.edges[(u, v)] = EdgeState(
                    u, v, self.cfg.default_bandwidth, self.cfg.default_prop_latency
                )

    def alive_nodes(self) -> List[int]:
        return [i for i, n in self.nodes.items() if not n.failed]

    def is_connected(self, extra_remove: Optional[Iterable[Edge]] = None, extra_add: Optional[Iterable[Edge]] = None) -> bool:
        g = self.G.copy()
        if extra_remove:
            g.remove_edges_from(extra_remove)
        if extra_add:
            g.add_edges_from(extra_add)
        failed = {i for i, n in self.nodes.items() if n.failed}
        g.remove_nodes_from(failed)
        if g.number_of_nodes() <= 1:
            return True
        return nx.is_connected(g)

    def degree(self, i: int) -> int:
        return int(self.G.degree(i)) if i in self.G else 0

    def neighbours(self, i: int) -> Set[int]:
        if i not in self.G:
            return set()
        return set(self.G.neighbors(i))

    def add_edge(self, u: int, v: int) -> bool:
        if u == v or self.nodes[u].failed or self.nodes[v].failed:
            return False
        e = canon(u, v)
        if self.G.has_edge(*e):
            return False
        self.G.add_edge(*e)
        self._sync_edges_from_graph()
        return True

    def remove_edge(self, u: int, v: int) -> bool:
        e = canon(u, v)
        if not self.G.has_edge(*e):
            return False
        self.G.remove_edge(*e)
        self._sync_edges_from_graph()
        return True

    def set_demand(self, matrix: np.ndarray) -> None:
        self.demand = np.asarray(matrix, dtype=float)
        np.fill_diagonal(self.demand, 0.0)

    def usable_graph(self) -> nx.Graph:
        """[SIM] Subgraph of alive nodes and non-failed edges — the topology as seen
        by routing and by the monitor. Failed elements stay in self.G (logical set)
        but are invisible here. [ASSUMPTION: indicators observe the usable topology]"""
        alive = {i for i, nd in self.nodes.items() if not nd.failed}
        g = nx.Graph()
        g.add_nodes_from(alive)
        for (u, v), e in self.edges.items():
            if not e.failed and u in alive and v in alive:
                g.add_edge(u, v)
        return g

    def clone_topology(self) -> nx.Graph:
        return self.G.copy()

    def restore_topology(self, g: nx.Graph) -> None:
        self.G = g.copy()
        self._sync_edges_from_graph()
