"""Constraint-aware, dependency-constrained joint mutation selector.

Modes
-----
* ``marginal`` — sort by the mutation-field score and accept greedily.
* ``pairwise`` — greedy with the original pairwise interaction bonus.
* ``joint`` (default when ``interaction_order >= 3``) — at each iteration pick
  the remaining candidate with the highest marginal **higher-order** score
  ``score + w_2 sum I_2 + w_3 sum I_3``, reject candidates whose conflict
  closure is incompatible with the chosen set, and optionally add a compatible
  second candidate when the joint higher-order value exceeds the best single.

The joint mode is a dependency-constrained greedy with a bounded look-ahead.
It is not an independent greedy: the compatibility structure can veto a
candidate that looks good marginally but conflicts with an already accepted
operation, and the third-order term can prefer a complementary group over the
best singleton.  A bounded local pass adds a compatible partner when the joint
value improves, which is the ``joint_local_search`` option.

Both modes apply the same budget, per-step cap and hard topology constraints.
Every rejected candidate is recorded with a machine-readable reason.

Complexity
----------
Let ``C`` be the number of feasible candidates, ``W`` the look-ahead window,
``k <= m_max`` the accepted count, and ``P`` the pair look-ahead width.

* pre-filter: ``O(|C|)`` cheap feasibility checks;
* each accepted mutation: ``O(W * k^2)`` higher-order score evaluations;
* pair look-ahead: ``O(P^2 * k^2)`` per accepted mutation (``P`` small);
* validation/application: ``O(k * (n+m))``;
* overall: ``O(|C| + k * (W + P^2) * k^2 + k(n+m))``.  With ``k`` and ``P``
  bounded constants this is ``O(|C| + W)`` selection work plus the topology
  validation cost.  Candidate generation itself remains ``O(n^2 + m k_r)``.

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
from plens.interaction import CompatibilityStructure, adjusted_score

_WINDOW = 64
_PAIR_WINDOW = 8


class _DegreeView:
    """Duck-typed view of ``net`` with memoised ``degree``/``neighbours``.

    The selector is the only topology writer inside a selection round, so the
    cache is invalidated explicitly after every accepted mutation or revert.
    This keeps the expensive ``networkx`` degree lookups at one per node per
    round instead of one per candidate comparison.
    """

    __slots__ = ("_net", "_deg", "_nbr")

    def __init__(self, net: EvolvingNetwork):
        self._net = net
        self._deg: Dict[int, int] = {}
        self._nbr: Dict[int, Set[int]] = {}

    def degree(self, i: int) -> int:
        d = self._deg.get(i)
        if d is None:
            d = self._net.degree(i)
            self._deg[i] = d
        return d

    def neighbours(self, i: int) -> Set[int]:
        s = self._nbr.get(i)
        if s is None:
            s = self._net.neighbours(i)
            self._nbr[i] = s
        return s

    def invalidate(self) -> None:
        self._deg.clear()
        self._nbr.clear()

    def __getattr__(self, name):
        return getattr(self._net, name)


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


def _reject_rest(remaining: List[dict], rejected: List[dict], reason: str) -> None:
    for c in sorted(remaining, key=lambda c: c["score"], reverse=True):
        rejected.append({**c, "reason": reason})


def _apply_one(
    net: EvolvingNetwork,
    view: _DegreeView,
    best: dict,
    cfg: Config,
    cooldown: Dict[Tuple[int, int], int],
    used_edges: Set[Tuple[int, int]],
    rejected: List[dict],
) -> tuple[bool, dict | None, float]:
    """Validate, apply and de-duplicate one candidate.  Returns (ok, chosen, cost)."""
    op = best["op"]
    cost = float(cfg.mutation_cost[op])
    ok, reason = muts.validate(net, best, cfg, cooldown)
    if not ok:
        rejected.append({**best, "reason": reason})
        return False, None, 0.0
    applied, detail = muts.apply(net, best)
    view.invalidate()
    if not applied:
        rejected.append({**best, "reason": "apply_failed"})
        return False, None, 0.0
    edge_key = canon(best["i"], best.get("l", best["j"]))
    if edge_key in used_edges:
        muts.revert_one(net, detail)
        view.invalidate()
        rejected.append({**best, "reason": "edge_already_used_this_step"})
        return False, None, 0.0
    used_edges.add(edge_key)
    if op == "REWIRE":
        used_edges.add(canon(best["i"], best["j"]))
    return True, {**best, "accepted": True, "reason": reason, "budget_cost": cost, "detail": detail}, cost


def _marginal_select(
    net: EvolvingNetwork,
    field: List[dict],
    budget: float,
    cfg: Config,
    cooldown: Dict[Tuple[int, int], int],
    interaction_weight: float,
) -> Tuple[List[dict], List[dict]]:
    chosen: List[dict] = []
    rejected: List[dict] = []
    view = _DegreeView(net)
    remaining = [c for c in field if _cheap_feasible(view, c, cfg, cooldown)]
    remaining.sort(key=lambda c: c["score"], reverse=True)
    spent = 0.0
    used_edges: Set[Tuple[int, int]] = set()
    min_cost = min(cfg.mutation_cost.values())

    while remaining and len(chosen) < cfg.max_mutations_per_step:
        if spent + min_cost > budget + 1e-9:
            _reject_rest(remaining, rejected, "insufficient_budget")
            break
        window = remaining[:_WINDOW]
        best = max(window, key=lambda c: adjusted_score(c, chosen, view, cfg, interaction_weight))
        remaining.remove(best)
        cost = float(cfg.mutation_cost[best["op"]])
        if spent + cost > budget + 1e-9:
            rejected.append({**best, "reason": "insufficient_budget"})
            continue
        ok, rec, cost = _apply_one(net, view, best, cfg, cooldown, used_edges, rejected)
        if not ok:
            continue
        spent += cost
        chosen.append(
            {
                **rec,
                "interaction_bonus": float(
                    adjusted_score(best, chosen[:-1], view, cfg, interaction_weight) - best["score"]
                ),
            }
        )
        remaining = [
            c
            for c in remaining
            if canon(c["i"], c.get("l", c["j"])) not in used_edges
            and _cheap_feasible(view, c, cfg, cooldown)
        ]
    return chosen, rejected


def _joint_select(
    net: EvolvingNetwork,
    field: List[dict],
    budget: float,
    cfg: Config,
    cooldown: Dict[Tuple[int, int], int],
    interaction_weight: float,
) -> Tuple[List[dict], List[dict]]:
    chosen: List[dict] = []
    rejected: List[dict] = []
    view = _DegreeView(net)
    compat = CompatibilityStructure(view, cfg)
    remaining = [c for c in field if _cheap_feasible(view, c, cfg, cooldown)]
    remaining.sort(key=lambda c: c["score"], reverse=True)
    spent = 0.0
    used_edges: Set[Tuple[int, int]] = set()
    min_cost = min(cfg.mutation_cost.values())
    window_size = max(int(getattr(cfg, "joint_window", _WINDOW)), 1)
    pair_window = min(max(int(getattr(cfg, "joint_pair_window", _PAIR_WINDOW)), 1), window_size)
    use_pairs = bool(getattr(cfg, "joint_local_search", True))

    # Defensive progress guard.  Every iteration must either accept a mutation,
    # reject/remove at least one candidate, or terminate; this bound only
    # protects against a future regression reintroducing a non-progress cycle.
    guard = len(remaining) + int(cfg.max_mutations_per_step) + 1
    while remaining and len(chosen) < cfg.max_mutations_per_step:
        guard -= 1
        if guard < 0:
            _reject_rest(remaining, rejected, "selection_guard")
            break
        if spent + min_cost > budget + 1e-9:
            _reject_rest(remaining, rejected, "insufficient_budget")
            break

        window = remaining[:window_size]

        # --- conflict propagation: drop candidates incompatible with the set ---
        conflict = [c for c in window if not compat.group_ok(chosen + [c])]
        for c in conflict:
            if c in remaining:
                remaining.remove(c)
            rejected.append({**c, "reason": "interaction_conflict"})
        window = [c for c in window if c not in conflict]
        if not window:
            continue

        # --- best singleton under higher-order marginal score ---
        best = max(window, key=lambda c: adjusted_score(c, chosen, view, cfg, interaction_weight))
        best_val = adjusted_score(best, chosen, view, cfg, interaction_weight)

        # --- optional compatible pair: genuine joint (not independent) move ---
        pair_choice = None
        if use_pairs and len(chosen) + 1 < cfg.max_mutations_per_step:
            pwindow = window[:pair_window]
            best_pair_val = -1e18
            best_pair = None
            for x in range(len(pwindow)):
                c1 = pwindow[x]
                for y in range(x + 1, len(pwindow)):
                    c2 = pwindow[y]
                    cost = float(cfg.mutation_cost[c1["op"]]) + float(cfg.mutation_cost[c2["op"]])
                    if spent + cost > budget + 1e-9:
                        continue
                    if not compat.group_ok(chosen + [c1, c2]):
                        continue
                    v1 = adjusted_score(c1, chosen, view, cfg, interaction_weight)
                    v2 = adjusted_score(c2, chosen + [c1], view, cfg, interaction_weight)
                    if min(v1, v2) <= 0.0:
                        continue
                    val = v1 + v2
                    if val > best_pair_val:
                        best_pair_val = val
                        best_pair = (c1, c2)
            margin = float(getattr(cfg, "joint_pair_gain_margin", 0.05))
            if best_pair is not None and best_pair_val > best_val + margin:
                pair_choice = best_pair

        # --- apply the chosen singleton or the compatible pair ---
        group = list(pair_choice) if pair_choice is not None else [best]
        for candidate in group:
            # Remove first: an unaffordable or invalid candidate must not stay
            # in ``remaining`` or the outer loop would select it forever.
            if candidate in remaining:
                remaining.remove(candidate)
            if len(chosen) >= cfg.max_mutations_per_step:
                break
            cost = float(cfg.mutation_cost[candidate["op"]])
            if spent + cost > budget + 1e-9:
                rejected.append({**candidate, "reason": "insufficient_budget"})
                continue
            ok, rec, cost = _apply_one(net, view, candidate, cfg, cooldown, used_edges, rejected)
            if not ok:
                continue
            spent += cost
            chosen.append(
                {
                    **rec,
                    "interaction_bonus": float(
                        adjusted_score(candidate, chosen[:-1], view, cfg, interaction_weight)
                        - candidate["score"]
                    ),
                }
            )
            remaining = [
                c
                for c in remaining
                if canon(c["i"], c.get("l", c["j"])) not in used_edges
                and _cheap_feasible(view, c, cfg, cooldown)
            ]

        # Refresh the compatibility view after the group was applied.
        remaining = [c for c in remaining if compat.group_ok(chosen + [c])]

    return chosen, rejected


def select_mutations(
    net: EvolvingNetwork,
    field: List[dict],
    budget: float,
    cfg: Config,
    cooldown: Dict[Tuple[int, int], int],
    interaction_weight: float | None = None,
    price: float | None = None,
    debt: list | None = None,
    uncertainty: list | None = None,
) -> Tuple[List[dict], List[dict]]:
    """Select and apply mutations under budget and hard topology constraints.

    ``price``/``debt``/``uncertainty`` are accepted for the unified optimizer;
    the field itself already carries the economically priced score, so they are
    used only for diagnostics/rejection metadata here.
    """
    weight = cfg.interaction_weight if interaction_weight is None else interaction_weight
    order = int(getattr(cfg, "interaction_order", 2))
    if order >= 3 or bool(getattr(cfg, "joint_local_search", False)):
        return _joint_select(net, field, budget, cfg, cooldown, weight)
    return _marginal_select(net, field, budget, cfg, cooldown, weight)
