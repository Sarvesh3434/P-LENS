"""[DIAGRAM] Observation indicators D, F, C, R — all per-node, clipped to [0, 1].

Every formula is an [ASSUMPTION], not a claim from an original paper.
"""

from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np

from network.graph import EvolvingNetwork, canon


def _jaccard_drift(curr, ref) -> float:
    inter = len(curr & ref)
    union = len(curr | ref)
    if union == 0:
        return 0.0
    return 1.0 - inter / union


def topology_drift(net: EvolvingNetwork) -> np.ndarray:
    """D_i = 0.5*(1 - Jaccard neighbourhood) + 0.5*degree relative change. [ASSUMPTION]

    Computed on the *usable* topology (failed elements excluded), because the
    monitor observes what routing actually sees."""
    g = net.usable_graph()
    d = np.zeros(net.n)
    for i in range(net.n):
        n_cur = set(g.neighbors(i)) if i in g else set()
        n_ref = set(net.reference.neighbors(i)) if i in net.reference else set()
        k_cur = len(n_cur)
        k_ref = len(n_ref)
        jac = _jaccard_drift(n_cur, n_ref)
        deg = abs(k_cur - k_ref) / max(k_cur, k_ref, 1)
        d[i] = 0.5 * jac + 0.5 * deg
    return np.clip(d, 0.0, 1.0)


def traffic_mismatch(net: EvolvingNetwork, demand_vec: np.ndarray) -> np.ndarray:
    """F_i = |d_i - p_i| / max(d_i, p_i, eps). [ASSUMPTION]"""
    f = np.zeros(net.n)
    for i, node in net.nodes.items():
        di = float(demand_vec[i])
        pi = float(node.provisioned)
        f[i] = abs(di - pi) / max(di, pi, 1e-9)
    return np.clip(f, 0.0, 1.0)


def congestion(net: EvolvingNetwork) -> np.ndarray:
    """C_i = w_u u_i + w_l max_link_util + w_q queue + w_p loss. [ASSUMPTION]"""
    w = net.cfg.congestion_weights
    c = np.zeros(net.n)
    for i, node in net.nodes.items():
        max_link = 0.0
        if i in net.G:
            for nb in net.G.neighbors(i):
                e = net.edges.get(canon(i, nb))
                if e:
                    max_link = max(max_link, min(e.utilization, 1.0))
        c[i] = (
            w["u"] * min(node.utilization, 1.0)
            + w["l"] * max_link
            + w["q"] * min(node.queue_delay, 1.0)
            + w["p"] * min(node.loss, 1.0)
        )
    return np.clip(c, 0.0, 1.0)


def resilience_deficiency(net: EvolvingNetwork) -> np.ndarray:
    """R_i = a*(1 - min(1, lambda_i/lambda_target)) + (1-a)*failure_impact. [ASSUMPTION]

    lambda_i is proxied by the usable-graph degree of node i [ASSUMPTION: cheap
    stand-in for local edge connectivity; documented, not exact]. Computed on the
    usable topology so failures immediately raise R_i."""
    a = 0.6
    target = net.cfg.lambda_target
    r = np.zeros(net.n)
    total_d = float(net.demand.sum()) + 1e-9
    g = net.usable_graph()
    for i in range(net.n):
        lam = float(g.degree(i)) if i in g else 0.0
        conn_term = 1.0 - min(1.0, lam / max(target, 1e-9))
        # failure_impact: fraction of demand involving i
        impact = (float(net.demand[i, :].sum()) + float(net.demand[:, i].sum())) / total_d
        r[i] = a * conn_term + (1.0 - a) * min(impact, 1.0)
        if net.nodes[i].failed:
            r[i] = 1.0
    return np.clip(r, 0.0, 1.0)


def compute_indicators(net: EvolvingNetwork, demand_for_f: np.ndarray) -> Dict[str, np.ndarray]:
    demand_vec = demand_for_f.sum(axis=1) + demand_for_f.sum(axis=0)
    return {
        "D": topology_drift(net),
        "F": traffic_mismatch(net, demand_vec),
        "C": congestion(net),
        "R": resilience_deficiency(net),
    }
