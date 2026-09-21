"""[DIAGRAM] Mutation field MF_ij^op per candidate edge. [ASSUMPTION]"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from experiments.config import Config
from network.graph import EvolvingNetwork, canon
from plens.budget import gate


def _need_debt(pp: np.ndarray, pd: np.ndarray, i: int, j: int) -> tuple[float, float]:
    return float(0.5 * (pp[i] + pp[j])), float(0.5 * (pd[i] + pd[j]))


def expected_benefit(net: EvolvingNetwork, i: int, j: int, op: str) -> float:
    """[ASSUMPTION] cheap proxy: demand between pair, congestion, under-connectivity."""
    dem = float(net.demand[i, j] + net.demand[j, i])
    tot = float(net.demand.sum()) + 1e-9
    cong = 0.5 * (net.nodes[i].utilization + net.nodes[j].utilization)
    deg_gap = max(0.0, 1.0 - net.degree(i) / max(net.cfg.max_degree, 1))
    if op == "ADD":
        return min(1.0, 4.0 * dem / tot + 0.5 * cong + 0.2 * deg_gap)
    if op == "PRUNE":
        idle = 1.0 - cong
        return min(1.0, idle * 0.6 + (1.0 - min(1.0, 8.0 * dem / tot)) * 0.4)
    # REWIRE: benefit if i is congested / mismatched
    return min(1.0, 0.5 * cong + 0.5 * deg_gap)


def constraint_penalty(net: EvolvingNetwork, i: int, j: int, op: str) -> float:
    pen = 0.0
    if net.nodes[i].failed or net.nodes[j].failed:
        pen += 1.0
    if op == "ADD" and net.G.has_edge(i, j):
        pen += 1.0
    if op == "ADD" and (net.degree(i) >= net.cfg.max_degree or net.degree(j) >= net.cfg.max_degree):
        pen += 0.8
    if op == "PRUNE" and not net.G.has_edge(i, j):
        pen += 1.0
    if op == "PRUNE" and (net.degree(i) <= net.cfg.min_degree or net.degree(j) <= net.cfg.min_degree):
        pen += 0.9
    return min(pen, 2.0)


def score_candidate(
    net: EvolvingNetwork,
    i: int,
    j: int,
    op: str,
    pp: np.ndarray,
    pd: np.ndarray,
    mc: Dict[tuple, float],
    budget: float,
    cfg: Config,
    use_credit: bool,
) -> float:
    a = cfg.field_weights
    need, debt = _need_debt(pp, pd, i, j)
    cred = mc.get(canon(i, j), 0.0) if use_credit else 0.0
    cost = cfg.mutation_cost[op] / max(max(cfg.mutation_cost.values()), 1e-9)
    g = gate(budget, cfg.mutation_cost[op])
    mf = g * (
        a["need"] * need
        + a["debt"] * debt
        + a["benefit"] * expected_benefit(net, i, j, op)
        + a["credit"] * cred
        - a["cost"] * cost
        - a["penalty"] * constraint_penalty(net, i, j, op)
    )
    return float(mf)


def generate_field(
    net: EvolvingNetwork,
    pp: np.ndarray,
    pd: np.ndarray,
    mc: Dict[tuple, float],
    budget: float,
    cfg: Config,
    use_credit: bool = True,
) -> List[dict]:
    """Enumerate ADD / PRUNE / REWIRE candidates and score them. Never random. [DIAGRAM]"""
    n = net.n
    out: List[dict] = []
    alive = net.alive_nodes()
    existing = {canon(u, v) for u, v in net.G.edges()}

    for a in alive:
        for b in alive:
            if a >= b:
                continue
            e = canon(a, b)
            if e not in existing:
                out.append(
                    {
                        "op": "ADD",
                        "i": a,
                        "j": b,
                        "score": score_candidate(net, a, b, "ADD", pp, pd, mc, budget, cfg, use_credit),
                    }
                )
    for u, v in existing:
        out.append(
            {
                "op": "PRUNE",
                "i": u,
                "j": v,
                "score": score_candidate(net, u, v, "PRUNE", pp, pd, mc, budget, cfg, use_credit),
            }
        )
        for l in alive:
            if l == u or l == v:
                continue
            if canon(u, l) in existing:
                continue
            # REWIRE: remove (u,v), add (u,l) — move one endpoint [ASSUMPTION from prompt]
            sc = score_candidate(net, u, l, "REWIRE", pp, pd, mc, budget, cfg, use_credit)
            out.append({"op": "REWIRE", "i": u, "j": v, "l": l, "score": sc})
    return out
