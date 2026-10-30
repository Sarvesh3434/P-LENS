"""Pairwise interaction model for joint mutation selection.

The mutation-field score is a *marginal* score.  Two individually good
operations can be redundant, mutually exclusive, or complementary.  We model a
pair ``(a,b)`` with a bounded interaction term ``I(a,b) in [-1,1]``:

* **Conflict (-1).**  The operations touch the same edge, one removes an edge
  the other adds, or both consume the last degree slot of a shared endpoint.
* **Redundancy (-0.4..-0.2).**  Two ADDs share an endpoint but the node has
  little headroom, or two PRUNEs would jointly threaten the same low-degree
  node.
* **Complementarity (+0.2..+0.7).**  Disjoint endpoints under congestion
  diversify paths; an ADD at a node paired with a PRUNE that frees a degree
  slot is synergistic; a REWIRE plus a disjoint ADD creates a new alternate
  route.

The selector then maximises ``score(c) + w_I * sum_{s in S} I(c,s)`` greedily,
subject to the same budget and hard constraints as the marginal selector.  This
is a lightweight, interpretable alternative to re-training a joint policy.
"""

from __future__ import annotations

from typing import Iterable

from experiments.config import Config
from network.graph import EvolvingNetwork, canon


def candidate_edges(c: dict) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Return (added, removed) canonical edges implied by a candidate."""
    op = c["op"]
    if op == "ADD":
        return {canon(int(c["i"]), int(c["j"]))}, set()
    if op == "PRUNE":
        return set(), {canon(int(c["i"]), int(c["j"]))}
    if op == "REWIRE":
        return {canon(int(c["i"]), int(c["l"]))}, {canon(int(c["i"]), int(c["j"]))}
    return set(), set()


def candidate_nodes(c: dict) -> set[int]:
    nodes = {int(c["i"]), int(c["j"])}
    if "l" in c:
        nodes.add(int(c["l"]))
    return nodes


def interaction(a: dict, b: dict, net: EvolvingNetwork, cfg: Config) -> float:
    add_a, rem_a = candidate_edges(a)
    add_b, rem_b = candidate_edges(b)
    nodes_a = candidate_nodes(a)
    nodes_b = candidate_nodes(b)
    shared = nodes_a & nodes_b

    # Exact edge conflicts.
    if add_a & rem_b or add_b & rem_a or add_a & add_b or rem_a & rem_b:
        return -1.0

    if shared:
        # Degrees are only needed to arbitrate operations that touch a common
        # node; computing them for disjoint pairs dominated the profile.
        headroom_a = min(cfg.max_degree - net.degree(i) for i in nodes_a)
        headroom_b = min(cfg.max_degree - net.degree(i) for i in nodes_b)
        degree_a = min(net.degree(i) for i in nodes_a)
        degree_b = min(net.degree(i) for i in nodes_b)

        # Both consume the last slot of a shared endpoint.
        if a["op"] == "ADD" and b["op"] == "ADD":
            if min(headroom_a, headroom_b) <= 1:
                return -1.0
            return -0.3

        # Pruning two edges incident to the same fragile node.
        if a["op"] == "PRUNE" and b["op"] == "PRUNE":
            if min(degree_a, degree_b) <= cfg.min_degree + 1:
                return -0.7
            return -0.2

        # ADD + PRUNE at the same node frees/consumes a degree slot: synergistic.
        if {a["op"], b["op"]} == {"ADD", "PRUNE"}:
            return 0.6

        # Any other pair sharing a node reinforces each other modestly.
        return 0.15

    # Disjoint pairs below.
    # Disjoint ADDs under congestion diversify the path set.
    if a["op"] == "ADD" and b["op"] == "ADD":
        cong = 0.5 * (
            net.nodes[int(a["i"])].utilization + net.nodes[int(a["j"])].utilization
            + net.nodes[int(b["i"])].utilization + net.nodes[int(b["j"])].utilization
        )
        return float(min(0.7, 0.2 + 0.5 * cong))

    # REWIRE + disjoint ADD: new alternate route.
    if {a["op"], b["op"]} == {"REWIRE", "ADD"}:
        return 0.3

    # Neighbouring operations can reinforce each other modestly.
    if any(j in net.neighbours(i) for i in nodes_a for j in nodes_b if i != j):
        return 0.15
    return 0.0


def adjusted_score(
    c: dict,
    chosen: Iterable[dict],
    net: EvolvingNetwork,
    cfg: Config,
    weight: float | None = None,
) -> float:
    w = cfg.interaction_weight if weight is None else weight
    bonus = sum(interaction(c, s, net, cfg) for s in chosen)
    return float(c["score"] + w * bonus)
