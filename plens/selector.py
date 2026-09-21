"""[DIAGRAM] Constraint-aware selector. Never picks at random.

Candidates are ranked by their mutation-field score and accepted in order while
the budget, the hard constraints and cfg.max_mutations_per_step allow. Every
rejected candidate is recorded with its reason; selections are applied
sequentially so later candidates see the earlier ones (no double-booking of an
edge or endpoint within a step).

Output per accepted mutation: op, i, j (, l), score, reason, budget_cost,
detail (what to add/remove — used for history, cooldown, reversal).
The rejected list feeds the audit trail: (candidate, reason).
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from experiments.config import Config
from network.graph import EvolvingNetwork, canon
from plens import mutations as muts


def select_mutations(
    net: EvolvingNetwork,
    field: List[dict],
    budget: float,
    cfg: Config,
    cooldown: Dict[Tuple[int, int], int],
) -> Tuple[List[dict], List[dict]]:
    ranked = sorted(field, key=lambda x: x["score"], reverse=True)
    chosen: List[dict] = []
    rejected: List[dict] = []
    spent = 0.0
    used_edges: Set[Tuple[int, int]] = set()

    for cand in ranked:
        if len(chosen) >= cfg.max_mutations_per_step:
            break
        op = cand["op"]
        cost = cfg.mutation_cost[op]
        if spent + cost > budget + 1e-9:
            rejected.append({**cand, "reason": "insufficient_budget"})
            continue
        ok, reason = muts.validate(net, cand, cfg, cooldown)
        if not ok:
            rejected.append({**cand, "reason": reason})
            continue
        applied, detail = muts.apply(net, cand)
        if not applied:
            rejected.append({**cand, "reason": "apply_failed"})
            continue
        edge_key = canon(cand["i"], cand.get("l", cand["j"]))
        if edge_key in used_edges:
            muts.revert_one(net, detail)
            rejected.append({**cand, "reason": "edge_already_used_this_step"})
            continue
        used_edges.add(edge_key)
        if op == "REWIRE":
            used_edges.add(canon(cand["i"], cand["j"]))
        spent += cost
        chosen.append(
            {
                **cand,
                "accepted": True,
                "reason": reason,
                "budget_cost": cost,
                "detail": detail,
                "score": cand["score"],
            }
        )
    # mutations stay applied: they ARE G_{t+1} [DIAGRAM: Evolving Network Topology]
    return chosen, rejected
