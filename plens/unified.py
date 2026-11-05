"""Unified P-LENS optimization.

All four mechanisms are special cases of one constrained programme.  At step
``t`` the controller chooses a feasible mutation set ``S`` maximising

    J(S) = sum_{a in S} [ v_a
                          - lambda_t c_a
                          + beta_D R_a
                          + beta_C MC_a
                          - rho u_a ]
           + w_2 sum_{a<b} I_2(a,b)
           + w_3 sum_{a<b<c} I_3(a,b,c)
           - mu churn(S)
           - nu risk(S)

    s.t.  c(S) <= B_t,  |S| <= m_max,
          degree / connectivity / lambda* constraints.

* ``v_a`` is the marginal structural need/benefit (pressure + expected benefit);
* ``lambda_t`` is the dual price from :mod:`plens.economics`, so the mutation
  economics term is the Lagrangian cost;
* ``beta_D R_a`` is the debt-relief potential of the mutation (plasticity debt);
* ``beta_C MC_a`` is the delayed counterfactual credit, itself a lower-
  confidence estimate from :mod:`plens.attribution`;
* ``u_a`` is the attribution uncertainty, so uncertain mutations are discounted;
* ``I_2`` and ``I_3`` are the dependency-aware pairwise/third-order interactions;
* ``churn(S)`` and ``risk(S)`` penalise topology instability and structural risk.

How each mechanism feeds the next decision
------------------------------------------
1. The monitor produces ``PP`` and the dependency-aware debt state ``PD``.
2. ``PD`` raises ``v_a`` and the debt-relief potential ``R_a``.
3. The economy sets ``lambda_t`` from the previous spend; it prices every
   candidate and caps the account ``B_t``.
4. The interaction structure filters hard conflicts and adds ``I_2/I_3`` to the
   joint value, so a pair/triple can beat the best singleton.
5. The selector applies the budget, per-step cap and feasibility constraints.
6. After execution, matched replay returns total/direct/indirect gain, its
   lower confidence bound, and a drift diagnostic.
7. The LCB updates counterfactual credit and debt relief; the economy updates
   ``lambda`` and ``B`` from spend and marginal utility.  The loop repeats.

Pseudocode
----------
```
for t = 0, 1, 2, ...:
    observe demand, failures; route and load
    PP_t   = EMA(weighted D,F,C,R)
    P_t    = dependency_matrix(topology, criticality)
    m_t    = modification_gate(recent mutations)
    PD_t   = update_debt(PD_{t-1}, PP_t, relief, P_t, m_t)
    lambda_t, B_t = economy_state
    C_t    = generate_candidates(PP_t, PD_t, MC, lambda_t, B_t)
    for c in C_t: score(c) = unified_score(c)
    S_t, rejected = joint_select(C_t, B_t, lambda_t, compatibility)
    execute S_t; log tokens and pre-failure snapshots
    for each due mutation a:
        (G_a, G_a^direct, G_a^indirect, LCB_a, drift_a) = matched_replay(a)
        if reliable and LCB_a passes gate:
            MC <- update_credit(MC, LCB_a * credit_weight)
            relief <- relief(a, LCB_a, confidence)
    PD_t <- update_debt(...)
    lambda_{t+1} = dual_ascent(lambda_t, spent_t, B_t)
    B_{t+1} = budget_account(B_t, spent_t, PD_t, LCB_t, MU_t, drift_t)
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from experiments.config import Config
from network.graph import EvolvingNetwork, canon
from plens import mutations as muts
from plens.mutation_field import constraint_penalty, expected_benefit
from plens.selector import select_mutations


@dataclass
class UnifiedState:
    pp: np.ndarray
    pd: np.ndarray
    mc: Dict[tuple, float]
    price: float
    budget: float
    uncertainty: float = 0.0
    debt_relief: np.ndarray | None = None
    churn: float = 0.0


def unified_score(
    net: EvolvingNetwork,
    c: dict,
    state: UnifiedState,
    cfg: Config,
) -> float:
    """Evaluate the linear part of the unified objective for one candidate."""
    i, j = int(c["i"]), int(c["j"])
    op = c["op"]
    a = cfg.field_weights
    need = 0.5 * (float(state.pp[i]) + float(state.pp[j]))
    debt = 0.5 * (float(state.pd[i]) + float(state.pd[j]))
    benefit = expected_benefit(net, i, j, op)
    credit = float(state.mc.get(canon(i, j), 0.0))
    cost = float(cfg.mutation_cost[op]) / max(max(cfg.mutation_cost.values()), 1e-9)
    relief = 0.0
    if state.debt_relief is not None:
        relief = 0.5 * (float(state.debt_relief[i]) + float(state.debt_relief[j]))
    risk = constraint_penalty(net, i, j, op)
    churn = 1.0 / max(int(cfg.max_mutations_per_step), 1)

    return float(
        a["need"] * need
        + a["debt"] * debt
        + a["benefit"] * benefit
        + float(getattr(cfg, "unified_credit_weight", a["credit"])) * credit
        - float(state.price) * cost
        - a["penalty"] * risk
        - float(getattr(cfg, "unified_uncertainty_weight", 0.0)) * max(float(state.uncertainty), 0.0)
        + float(getattr(cfg, "unified_debt_weight", 0.0)) * relief
        - float(getattr(cfg, "unified_churn_weight", 0.0)) * churn
        - float(getattr(cfg, "unified_risk_weight", 0.0)) * risk
    )


def apply_unified_scores(
    net: EvolvingNetwork,
    field: List[dict],
    state: UnifiedState,
    cfg: Config,
) -> List[dict]:
    for c in field:
        c["score"] = unified_score(net, c, state, cfg)
    return field


def unified_select(
    net: EvolvingNetwork,
    field: List[dict],
    state: UnifiedState,
    cfg: Config,
    cooldown: Dict[Tuple[int, int], int],
    interaction_weight: float | None = None,
) -> Tuple[List[dict], List[dict]]:
    """Score with the unified objective and run dependency-constrained selection."""
    scored = apply_unified_scores(net, field, state, cfg)
    return select_mutations(
        net,
        scored,
        state.budget,
        cfg,
        cooldown,
        interaction_weight=cfg.interaction_weight if interaction_weight is None else interaction_weight,
        price=state.price,
    )


def unified_pseudocode() -> str:
    """Return the paper-ready pseudocode block."""
    return __doc__.split("Pseudocode\n----------\n", 1)[-1].strip("`\n ")
