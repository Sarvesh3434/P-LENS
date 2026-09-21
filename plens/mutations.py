"""[DIAGRAM] ADD / PRUNE / REWIRE. Independently unit-testable. [ASSUMPTION] REWIRE moves one endpoint."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import networkx as nx

from experiments.config import Config
from network.graph import EvolvingNetwork, canon


def validate(net: EvolvingNetwork, cand: dict, cfg: Config, cooldown: Dict[Tuple[int, int], int]) -> tuple[bool, str]:
    op = cand["op"]
    i, j = int(cand["i"]), int(cand["j"])
    if i == j:
        return False, "self_loop"
    if net.nodes[i].failed or net.nodes[j].failed:
        return False, "endpoint_failed"
    cd_key = canon(i, j)
    if cooldown.get(cd_key, 0) > 0:
        return False, "cooldown"

    if op == "ADD":
        if net.G.has_edge(i, j):
            return False, "already_exists"
        if net.degree(i) >= cfg.max_degree or net.degree(j) >= cfg.max_degree:
            return False, "degree_cap"
        return True, "ok_add"

    if op == "PRUNE":
        if not net.G.has_edge(i, j):
            return False, "missing_edge"
        # hard safety constraint (never disconnect) is checked before the soft
        # degree bound, so a ring-edge prune reports would_disconnect [DIAGRAM]
        if cfg.connectivity_required and not net.is_connected(extra_remove=[(i, j)]):
            return False, "would_disconnect"
        # global resilience target [DIAGRAM λ ≥ λ_target]: the usable network's
        # average degree must stay at or above lambda_target; a prune that would
        # undercut it is refused for the same reason as a disconnection [ASSUMPTION]
        alive = len(net.alive_nodes())
        m_usable = sum(1 for e in net.edges.values() if not e.failed)
        e_ij = net.edges.get(canon(i, j))
        prunable = 0 if (e_ij is not None and e_ij.failed) else 1
        if alive > 1 and (2.0 * (m_usable - prunable)) / alive < cfg.lambda_target:
            return False, "would_disconnect"
        if net.degree(i) <= cfg.min_degree or net.degree(j) <= cfg.min_degree:
            return False, "min_degree"
        return True, "ok_prune"

    if op == "REWIRE":
        l = int(cand.get("l", -1))
        if l < 0 or l == i or l == j:
            return False, "bad_rewire_target"
        if net.nodes[l].failed:
            return False, "new_endpoint_failed"
        if not net.G.has_edge(i, j):
            return False, "missing_edge"
        if net.G.has_edge(i, l):
            return False, "already_exists"
        if net.degree(l) >= cfg.max_degree:
            return False, "degree_cap"
        # node j loses edge (i, j): its post-state degree must respect min_degree
        if net.degree(j) <= cfg.min_degree:
            return False, "min_degree"
        if cfg.connectivity_required and not net.is_connected(
            extra_remove=[(i, j)], extra_add=[(i, l)]
        ):
            return False, "would_disconnect"
        if cooldown.get(canon(i, l), 0) > 0:
            return False, "cooldown"
        return True, "ok_rewire"

    return False, "unknown_op"


def apply(net: EvolvingNetwork, cand: dict) -> tuple[bool, dict]:
    op = cand["op"]
    i, j = int(cand["i"]), int(cand["j"])
    if op == "ADD":
        ok = net.add_edge(i, j)
        return ok, {"op": "ADD", "added": [i, j], "removed": None}
    if op == "PRUNE":
        ok = net.remove_edge(i, j)
        return ok, {"op": "PRUNE", "added": None, "removed": [i, j]}
    if op == "REWIRE":
        l = int(cand["l"])
        if not net.remove_edge(i, j):
            return False, {}
        if not net.add_edge(i, l):
            net.add_edge(i, j)
            return False, {}
        return True, {"op": "REWIRE", "added": [i, l], "removed": [i, j]}
    return False, {}


def revert_one(net: EvolvingNetwork, detail: dict) -> None:
    if not detail:
        return
    if detail.get("added"):
        a, b = detail["added"]
        net.remove_edge(a, b)
    if detail.get("removed"):
        a, b = detail["removed"]
        net.add_edge(a, b)


def reverse_of(detail: dict) -> dict:
    """For counterfactual shadow evaluation: undo this mutation."""
    return {
        "op": "REVERSE",
        "added": detail.get("removed"),
        "removed": detail.get("added"),
    }
