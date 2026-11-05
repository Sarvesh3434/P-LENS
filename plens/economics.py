"""Mutation economics: a primal-dual interpretation of the adaptive budget.

Constrained topology-mutation programme
---------------------------------------
At step ``t`` the controller solves

    max_{S in F_t}   V(S)
    s.t.             c(S) <= B_t,          |S| <= m_max,
                     structural feasibility (degree, connectivity, lambda*)

where ``c(S) = sum_{a in S} c_a`` is the mutation cost, ``B_t`` a bounded
resource account (the number of cost units the controller may spend this step),
``V`` the estimated long-term network value, and ``F_t`` the feasible mutation
sets.  Introduce the multiplier ``lambda_t >= 0`` for the resource constraint:

    L(S, lambda_t) = V(S) - lambda_t (c(S) - B_t)
                   = V(S) - lambda_t c(S) + lambda_t B_t.

For fixed ``B_t`` the last term is constant, so the primal selection rule is

    S_t = argmax_{S in F_t, c(S) <= B_t} [ V(S) - lambda_t c(S) ].

The dual function is ``g(lambda) = max_{S in F_t} L(S, lambda)``.  The dual
price obeys the projected ascent

    lambda_{t+1} = Proj_[lambda_min, lambda_max]
                    ( lambda_t + alpha_t ( c(S_t) - rho B_t ) / max(B_t, c_min) )

with target spend fraction ``rho`` and step ``alpha_t = alpha_0 / sqrt(1+t)``.
A price above zero means the resource constraint binds: one more cost unit is
worth ``lambda_t`` units of value.  At the optimum the marginal condition is

    marginal value of a mutation  =  lambda_t * marginal cost,

which is exactly how the selector prices a candidate.  ``B_t`` is *not* itself
the dual variable; it is the bounded primal resource account whose scarcity is
priced by ``lambda_t``.  This distinction is stated explicitly because the
original draft conflated the two.

Budget account update
---------------------
The account is a bounded debit/credit resource driven by three economically
interpretable terms:

    LCB_t      = mean_gain_t - z * std_gain_t
    safety_t   = 1 / (1 + a_o overload_t + a_f failure_t)
    damping_t  = (1 - 0.5 uncertainty_t) * safety_t * refill_scale_t
    refill_t   = (b_0 + beta PD_t) * damping_t / (1 + d_p lambda_t)
    credit_t   = (eta_g LCB_t - eta_s churn_t) * damping_t
    MU_t       = (V_hat_t - lambda_t c_t) / max(c_t, c_min)
    B_{t+1}    = Proj_[B_min, B_max]( B_t - c_t + refill_t + credit_t
                                       + kappa_mu * tanh(max(MU_t - MU_0, 0)) * damping_t )

* ``refill`` is the debt-justified stipend.
* ``credit`` is the measured (lower-confidence) value of past mutations minus
  the churn price.
* ``MU`` is the marginal utility of the last mutation; it raises the account
  only when a mutation produced value above its dual price, which prevents
  unnecessary suppression of beneficial mutations.
* the dual price ``lambda`` rises when spending saturates the account, which
  prevents excessive topology changes; it falls when the account is idle, which
  prevents over-suppression.
* projection onto ``[B_min, B_max]`` gives boundedness by construction.

Stability / regret (informal statement; full assumptions in :mod:`plens.theory`)
-------------------------------------------------------------------------------
Assume (i) ``V`` is concave in the (convexified) selection variable, (ii) the
gradient of the Lagrangian is bounded by ``G``, (iii) Slater's condition holds,
and (iv) ``alpha_t = alpha_0 / sqrt(t)``.  Then projected primal-dual ascent
achieves ``O(sqrt(T))`` dual regret and the averaged primal iterate converges to
the feasible region at rate ``O(1/sqrt(T))``.  The bounded account adds only the
projection error ``O(1/T)``.  These are standard projected subgradient results;
the code exposes the assumptions as executable checks in :mod:`plens.theory`.

``mode="legacy"`` reproduces the original linear update for comparison;
``mode="fixed"`` pins the budget to its initial value; ``enabled=False`` is the
``no_budget`` ablation and pins it to ``B_max``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.config import Config


@dataclass
class EconomyState:
    budget: float
    price: float
    spent: float = 0.0
    refill: float = 0.0
    credit: float = 0.0
    marginal_utility: float = 0.0
    dual_residual: float = 0.0
    budget_target: float = 0.0
    price_step: float = 0.0


def project(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def lagrangian(
    value: float,
    cost: float,
    price: float,
    budget: float = 0.0,
    churn: float = 0.0,
    risk: float = 0.0,
    churn_penalty: float = 0.0,
    risk_penalty: float = 0.0,
) -> float:
    """``L(S, lambda) = V - lambda (c - B) - mu churn - nu risk``."""
    return float(
        value
        - price * (cost - budget)
        - churn_penalty * churn
        - risk_penalty * risk
    )


def marginal_utility(value: float, cost: float, price: float, c_min: float = 1e-9) -> float:
    """Net value per unit cost: ``(V_hat - lambda c) / max(c, c_min)``.

    Positive means the mutation's estimated value exceeds its dual price and is
    therefore worth funding; negative means the account should not be expanded
    for it.
    """
    return float((value - price * cost) / max(cost, c_min))


def dual_residual(
    spent: float,
    budget: float,
    cfg: Config,
    churn: float = 0.0,
    c_min: float | None = None,
) -> float:
    """Normalised violation of the target spend fraction."""
    target = float(getattr(cfg, "budget_target_utilization", 0.75))
    scale = max(float(budget), c_min or min(cfg.mutation_cost.values()), 1e-9)
    residual = (spent - target * budget) / scale
    residual += float(getattr(cfg, "budget_churn_penalty", 0.0)) * max(churn, 0.0)
    return float(residual)


def update_dual_price(
    price: float,
    spent: float,
    budget: float,
    cfg: Config,
    iteration: int = 0,
    churn: float = 0.0,
    step: float | None = None,
) -> float:
    """Projected dual ascent on the mutation-resource multiplier."""
    residual = dual_residual(spent, budget, cfg, churn=churn)
    if step is None:
        alpha0 = float(getattr(cfg, "budget_dual_step", 0.08))
        step = alpha0 / np.sqrt(1.0 + max(int(iteration), 0))
    # Forgetting keeps the price responsive to a changing environment without
    # erasing the accumulated scarcity signal.
    decay = float(getattr(cfg, "budget_dual_decay", 0.995))
    scale = float(getattr(cfg, "budget_price_scale", 1.0))
    nxt = decay * float(price) + float(step) * scale * residual
    return project(
        nxt,
        float(getattr(cfg, "budget_price_min", 0.0)),
        float(getattr(cfg, "budget_price_max", 3.0)),
    )


def _budget_refill(
    budget: float,
    spent: float,
    mean_pd: float,
    mean_gain: float,
    mean_stability: float,
    price: float,
    mu: float,
    cfg: Config,
    gain_std: float,
    overload: float,
    failure_rate: float,
    uncertainty: float,
    refill_scale: float,
) -> tuple[float, float, float, float]:
    lcb = float(mean_gain) - float(getattr(cfg, "budget_uncertainty_z", 1.0)) * max(float(gain_std), 0.0)
    safety = 1.0 / (
        1.0
        + float(getattr(cfg, "budget_overload_damping", 0.0)) * max(float(overload), 0.0)
        + float(getattr(cfg, "budget_failure_damping", 0.0)) * max(float(failure_rate), 0.0)
    )
    damping = (
        (1.0 - 0.5 * float(np.clip(uncertainty, 0.0, 1.0)))
        * max(float(refill_scale), 0.0)
        * safety
    )
    price_damp = 1.0 / (
        1.0 + float(getattr(cfg, "budget_price_refill_damping", 0.0)) * max(price, 0.0)
    )
    refill = (
        float(getattr(cfg, "budget_base_refill", 0.0))
        + float(getattr(cfg, "budget_refill", 0.0)) * max(float(mean_pd), 0.0)
    ) * damping * price_damp
    credit = (
        float(getattr(cfg, "eta_g", 0.5)) * lcb
        - float(getattr(cfg, "eta_s", 0.4)) * max(float(mean_stability), 0.0)
    ) * damping
    mu_ref = float(getattr(cfg, "budget_marginal_ref", 0.05))
    mu_bonus = (
        float(getattr(cfg, "budget_marginal_gain", 0.0))
        * float(np.tanh(max(float(mu) - mu_ref, 0.0)))
        * damping
    )
    # Persistent overload/failure halves the stipend but does not freeze the
    # account: spending still costs, so churn decays when actions stop helping.
    if overload >= float(getattr(cfg, "budget_overload_threshold", 1e9)) or failure_rate >= 0.5:
        refill *= 0.5
    nxt = float(budget) - float(spent) + refill + credit + mu_bonus
    return nxt, refill, credit, mu_bonus


def economy_step(
    state: EconomyState,
    spent: float,
    mean_pd: float,
    mean_gain: float,
    mean_stability: float,
    cfg: Config,
    enabled: bool = True,
    gain_std: float = 0.0,
    overload: float = 0.0,
    failure_rate: float = 0.0,
    uncertainty: float = 0.0,
    refill_scale: float = 1.0,
    mode: str | None = None,
    iteration: int = 0,
    churn: float = 0.0,
) -> EconomyState:
    """One primal-dual step of the mutation economy."""
    if not enabled:
        return EconomyState(budget=float(cfg.mutation_budget_max), price=float(state.price))

    mode = mode or str(getattr(cfg, "budget_mode", "adaptive"))
    if mode == "fixed":
        return EconomyState(
            budget=float(cfg.mutation_budget_initial),
            price=float(state.price),
            spent=float(spent),
        )

    if mode == "legacy":
        nxt = (
            float(state.budget)
            - float(spent)
            + float(getattr(cfg, "budget_refill", 0.0)) * max(float(mean_pd), 0.0)
            + float(getattr(cfg, "eta_g", 0.5)) * float(mean_gain)
            - float(getattr(cfg, "eta_s", 0.4)) * float(mean_stability)
        )
        return EconomyState(
            budget=project(nxt, cfg.mutation_budget_min, cfg.mutation_budget_max),
            price=float(state.price),
            spent=float(spent),
        )

    mu = marginal_utility(float(mean_gain), max(float(spent), 1e-9), float(state.price))
    price = update_dual_price(
        state.price, spent, state.budget, cfg, iteration=iteration, churn=churn
    )
    nxt, refill, credit, mu_bonus = _budget_refill(
        state.budget,
        spent,
        mean_pd,
        mean_gain,
        mean_stability,
        price,
        mu,
        cfg,
        gain_std,
        overload,
        failure_rate,
        uncertainty,
        refill_scale,
    )
    return EconomyState(
        budget=project(nxt, cfg.mutation_budget_min, cfg.mutation_budget_max),
        price=price,
        spent=float(spent),
        refill=float(refill),
        credit=float(credit),
        marginal_utility=float(mu),
        dual_residual=float(dual_residual(spent, state.budget, cfg, churn=churn)),
        budget_target=float(getattr(cfg, "budget_target_utilization", 0.75) * state.budget),
        price_step=float(price - state.price),
    )
