"""Higher-order interaction model for joint mutation selection.

The mutation-field score is a *marginal* score.  Two or three individually good
operations can be redundant, mutually exclusive, complementary, or jointly
destabilising.  We model a set ``S`` of operations with a bounded set function

    Phi(S) = sum_{a in S} score_a
             + w_2 * sum_{a<b in S} I_2(a,b)
             + w_3 * sum_{a<b<c in S} I_3(a,b,c)

with ``I_2, I_3 in [-1, 1]``.  The pairwise term is the original interaction
model.  The third-order term captures effects that cannot be expressed by pairs:

* **slot competition.**  Three operations that all consume the last degree slot
  of a shared endpoint are jointly infeasible even if every pair looks benign.
* **conflict propagation.**  If one pair is a hard conflict, every triple that
  contains it inherits the conflict; the compatibility structure closes the
  conflict relation transitively before selection.
* **synergy.**  ADD + PRUNE + REWIRE on a connected neighbourhood can free a
  slot, add an alternate path and relocate an endpoint in one joint move.
* **destabilisation.**  Three PRUNEs around a fragile low-degree node, or a
  group that jointly threatens connectivity, receive a negative third-order
  factor even when each pair is only mildly negative.
* **path diversity.**  Disjoint ADDs under congestion have diminishing
  individual returns; the triple factor saturates their combined benefit.

The compatibility structure exposes three relations for a group: positive
(complementary), redundant (diminishing returns), and destabilising (negative
or infeasible).  Selection is dependency-constrained: a candidate that hard-
conflicts with the chosen set is rejected with ``reason="interaction_conflict"``
instead of being silently out-ranked.

Complexity
----------
* candidate generation: ``O(n^2 + m k)`` for ADD/PRUNE/REWIRE (``k`` REWIRE
  targets per edge);
* pairwise scoring of one candidate against a chosen set: ``O(|S|)``;
* third-order scoring: ``O(|S|^2)`` per candidate, and ``|S| <= m_max`` is a
  small constant (default 4);
* joint greedy selection with a look-ahead window ``W``: ``O(k * W * m_max^2)``
  score evaluations plus ``O(k * (n+m))`` feasibility checks, where ``k`` is
  the number of accepted mutations;
* bounded local search: ``O(L * m_max^2 * W)`` for ``L`` improvement passes.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

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


def _slot_delta(c: dict, v: int) -> int:
    """Degree-slot change at node ``v`` implied by candidate ``c``."""
    i, j = int(c["i"]), int(c["j"])
    op = c["op"]
    if op == "ADD":
        return 1 if v in (i, j) else 0
    if op == "PRUNE":
        return -1 if v in (i, j) else 0
    if op == "REWIRE":
        l = int(c.get("l", -1))
        d = 0
        if v == l:
            d += 1
        if v == j:
            d -= 1
        return d
    return 0


def _congestion(net: EvolvingNetwork, nodes: Iterable[int]) -> float:
    vals = [float(net.nodes[v].utilization) for v in nodes if v in net.nodes]
    return float(np.mean(vals)) if vals else 0.0


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


def hard_conflict(a: dict, b: dict, net: EvolvingNetwork, cfg: Config) -> bool:
    """True when two operations cannot coexist (conflict closure primitive)."""
    return interaction(a, b, net, cfg) <= -0.999


def interaction_triple(
    a: dict,
    b: dict,
    c: dict,
    net: EvolvingNetwork,
    cfg: Config,
) -> float:
    """Bounded third-order interaction ``I_3(a,b,c)``.

    The factor is the ``tanh`` of the pairwise evidence plus explicit three-way
    slot, synergy and destabilisation terms.  ``tanh`` keeps it in ``(-1, 1)``
    and makes the marginal effect of additional operations saturate.
    """
    p_ab = interaction(a, b, net, cfg)
    p_ac = interaction(a, c, net, cfg)
    p_bc = interaction(b, c, net, cfg)
    pair_sum = p_ab + p_ac + p_bc

    # Conflict propagation: a hard conflict anywhere makes the triple infeasible.
    if min(p_ab, p_ac, p_bc) <= -0.999:
        return -1.0

    group = (a, b, c)
    ops = {x["op"] for x in group}
    nodes = set().union(*(candidate_nodes(x) for x in group))
    shared = (candidate_nodes(a) & candidate_nodes(b)) | (
        candidate_nodes(a) & candidate_nodes(c)
    ) | (candidate_nodes(b) & candidate_nodes(c))

    h = 0.0

    # 1. Joint degree-slot competition at shared endpoints.
    for v in shared:
        headroom = int(cfg.max_degree - net.degree(v))
        demand_slots = sum(max(0, _slot_delta(x, v)) for x in group)
        if demand_slots > headroom:
            h -= 0.60 * (demand_slots - headroom)

    # 2. Joint destabilisation: pruning around a fragile node.
    for v in nodes:
        degree = int(net.degree(v))
        pruning = sum(1 for x in group if x["op"] == "PRUNE" and v in candidate_nodes(x))
        if pruning and degree - pruning < cfg.min_degree:
            h -= 0.45 * pruning
        if pruning >= 2 and degree <= cfg.min_degree + 1:
            h -= 0.30

    # 3. Three-way synergy: free a slot, add capacity, relocate an endpoint.
    if ops == {"ADD", "PRUNE", "REWIRE"} and shared:
        h += 0.25
        if _congestion(net, nodes) > 0.5:
            h += 0.10

    # 4. Path diversity for three disjoint ADDs: saturating positive factor.
    if ops == {"ADD"} and len(shared) == 0:
        h += 0.20 * min(1.0, _congestion(net, nodes))

    # 5. Repeated PRUNEs are jointly redundant when they touch a common node.
    if ops == {"PRUNE"} and shared:
        h -= 0.20

    # 6. A joint group that would leave a shared node at the degree floor is
    #    structurally risky even if pairwise scores are non-negative.
    for v in shared:
        net_delta = sum(_slot_delta(x, v) for x in group)
        if net.degree(v) + net_delta <= cfg.min_degree:
            h -= 0.35

    return float(np.clip(np.tanh(pair_sum + h), -1.0, 1.0))


def group_effect(
    group: Sequence[dict],
    net: EvolvingNetwork,
    cfg: Config,
    weight2: float | None = None,
    weight3: float | None = None,
) -> float:
    """Total interaction value ``Phi(S) - sum score`` for a group."""
    if len(group) < 2:
        return 0.0
    w2 = float(cfg.interaction_weight if weight2 is None else weight2)
    w3 = float(getattr(cfg, "interaction_triple_weight", 0.0) if weight3 is None else weight3)
    order = int(getattr(cfg, "interaction_order", 2))
    total = 0.0
    for x in range(len(group)):
        for y in range(x + 1, len(group)):
            total += w2 * interaction(group[x], group[y], net, cfg)
    if order >= 3:
        for x in range(len(group)):
            for y in range(x + 1, len(group)):
                for z in range(y + 1, len(group)):
                    total += w3 * interaction_triple(group[x], group[y], group[z], net, cfg)
    return float(total)


def higher_order_bonus(
    c: dict,
    chosen: Iterable[dict],
    net: EvolvingNetwork,
    cfg: Config,
    weight: float | None = None,
) -> float:
    """Marginal interaction bonus of adding ``c`` to ``chosen``."""
    chosen = list(chosen)
    w2 = float(cfg.interaction_weight if weight is None else weight)
    order = int(getattr(cfg, "interaction_order", 2))
    if order <= 1 or not chosen or abs(w2) <= 1e-12:
        return 0.0
    bonus = sum(interaction(c, s, net, cfg) for s in chosen) * w2
    if order >= 3:
        w3 = float(getattr(cfg, "interaction_triple_weight", 0.0))
        for x in range(len(chosen)):
            for y in range(x + 1, len(chosen)):
                bonus += w3 * interaction_triple(c, chosen[x], chosen[y], net, cfg)
    return float(bonus)


def adjusted_score(
    c: dict,
    chosen: Iterable[dict],
    net: EvolvingNetwork,
    cfg: Config,
    weight: float | None = None,
) -> float:
    return float(c["score"] + higher_order_bonus(c, chosen, net, cfg, weight))


class CompatibilityStructure:
    """Dependency-constrained compatibility of mutation groups.

    The structure answers the three questions the selector needs:

    * ``relation(a, b)``: positive / redundant / destabilising / conflict;
    * ``group_ok(group)``: whether the group has a hard conflict closure;
    * ``group_effect(group)``: cumulative interaction value, including the
      third-order resilience/destabilisation factors.
    """

    def __init__(self, net: EvolvingNetwork, cfg: Config):
        self.net = net
        self.cfg = cfg

    def relation(self, a: dict, b: dict) -> str:
        v = interaction(a, b, self.net, self.cfg)
        if v <= -0.999:
            return "conflict"
        if v <= -0.15:
            return "redundant"
        if v >= 0.15:
            return "synergy"
        return "neutral"

    def group_ok(self, group: Sequence[dict]) -> bool:
        if not bool(getattr(self.cfg, "joint_conflict_propagation", True)):
            return True
        for x in range(len(group)):
            for y in range(x + 1, len(group)):
                if hard_conflict(group[x], group[y], self.net, self.cfg):
                    return False
        if int(getattr(self.cfg, "interaction_order", 2)) >= 3:
            for x in range(len(group)):
                for y in range(x + 1, len(group)):
                    for z in range(y + 1, len(group)):
                        if interaction_triple(
                            group[x], group[y], group[z], self.net, self.cfg
                        ) <= -0.999:
                            return False
        return True

    def group_effect(self, group: Sequence[dict]) -> float:
        return group_effect(group, self.net, self.cfg)

    def cumulative_resilience_impact(self, group: Sequence[dict]) -> float:
        """Signed resilience proxy for a group in ``[-1, 1]``.

        Positive means the group adds usable capacity / path diversity; negative
        means it consumes degree headroom or threatens low-degree nodes.
        """
        net = self.net
        score = 0.0
        for c in group:
            for v in candidate_nodes(c):
                delta = _slot_delta(c, v)
                headroom = max(1.0, float(self.cfg.max_degree - net.degree(v)))
                score += delta / headroom
        score /= max(len(group), 1)
        return float(np.clip(np.tanh(score), -1.0, 1.0))
