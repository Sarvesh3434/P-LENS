"""Constraint-aware, interaction-aware mutation selector.

Two modes are supported:

* ``marginal`` — sort by the mutation-field score and accept greedily.
* ``interaction`` (default) — at each iteration pick the remaining candidate
  with the highest ``score + w_I * sum_{s in S} I(c,s)``, where ``I`` is the
  pairwise interaction model in :mod:`plens.interaction`.  This lets the
  controller choose complementary pairs (e.g. an ADD on a congested pair plus a
  PRUNE that frees a degree slot) and avoid redundant or conflicting ones.

Both modes apply the same budget, per-step cap and hard topology constraints.
Every rejected candidate is recorded with a machine-readable reason.

Performance note: the field contains O(n^2) candidates and most are infeasible
at a given topology.  We pre-filter cheap constraints and re-rank only a small
top window after each accepted mutation, instead of calling the interaction
model for every candidate on every iteration.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from experiments.config import Config
from network.graph import EvolvingNetwork, canon
from plens import mutations as muts
from plens.interaction import adjusted_score

_WINDOW = 64


def _cheap_feasible(
    net: EvolvingNetwork,
    c: dict,
    cfg: Config,
    cooldown: Dict[Tuple[int, int], int],
) -> bool:
    i, j = int(c["i"]), int(c["j"])
    if i == j or net.nodes[i].failed or net.nodes[j].failed:
        return False
    op = c["op"]
    if op in ("ADD", "PRUNE") and cooldown.get(canon(i, j), 0) > 0:
        return False
    if op == "ADD":
        if net.G.has_edge(i, j):
            return False
        if net.degree(i) >= cfg.max_degree or net.degree(j) >= cfg.max_degree:
            return False
        return True
    if op == "PRUNE":
        if not net.G.has_edge(i, j):
            return False
        if net.degree(i) <= cfg.min_degree or net.degree(j) <= cfg.min_degree:
            return False
        return True
    if op == "REWIRE":
        l = int(c.get("l", -1))
        if l < 0 or l == i or l == j:
            return False
        if net.nodes[l].failed or cooldown.get(canon(i, l), 0) > 0:
            return False
        if not net.G.has_edge(i, j) or net.G.has_edge(i, l):
            return False
        if net.degree(l) >= cfg.max_degree or net.degree(j) <= cfg.min_degree:
            return False
        return True
    return False


def select_mutations(
    net: EvolvingNetwork,
    field: List[dict],
    budget: float,
    cfg: Config,
    cooldown: Dict[Tuple[int, int], int],
    interaction_weight: float | None = None,
) -> Tuple[List[dict], List[dict]]:
    chosen: List[dict] = []
    rejected: List[dict] = []
    remaining = [c for c in field if _cheap_feasible(net, c, cfg, cooldown)]
    remaining.sort(key=lambda c: c["score"], reverse=True)
    spent = 0.0
    used_edges: Set[Tuple[int, int]] = set()
    weight = cfg.interaction_weight if interaction_weight is None else interaction_weight

    while remaining and len(chosen) < cfg.max_mutations_per_step:
        window = remaining[:_WINDOW]
        best = max(window, key=lambda c: adjusted_score(c, chosen, net, cfg, weight))
        remaining.remove(best)

        op = best["op"]
        cost = cfg.mutation_cost[op]
        if spent + cost > budget + 1e-9:
            rejected.append({**best, "reason": "insufficient_budget"})
            continue
        ok, reason = muts.validate(net, best, cfg, cooldown)
        if not ok:
            rejected.append({**best, "reason": reason})
            continue
        applied, detail = muts.apply(net, best)
        if not applied:
            rejected.append({**best, "reason": "apply_failed"})
            continue
        edge_key = canon(best["i"], best.get("l", best["j"]))
        if edge_key in used_edges:
            muts.revert_one(net, detail)
            rejected.append({**best, "reason": "edge_already_used_this_step"})
            continue
        used_edges.add(edge_key)
        if op == "REWIRE":
            used_edges.add(canon(best["i"], best["j"]))
        spent += cost
        chosen.append(
            {
                **best,
                "accepted": True,
                "reason": reason,
                "budget_cost": cost,
                "detail": detail,
                "interaction_bonus": float(
                    adjusted_score(best, chosen[:-1], net, cfg, weight) - best["score"]
                ),
            }
        )
        # Topology changed: cheap constraints and consumed edges must be refreshed.
        remaining = [
            c
            for c in remaining
            if canon(c["i"], c.get("l", c["j"])) not in used_edges
            and _cheap_feasible(net, c, cfg, cooldown)
        ]
    # mutations stay applied: they ARE G_{t+1} [DIAGRAM: Evolving Network Topology]
    return chosen, rejected
