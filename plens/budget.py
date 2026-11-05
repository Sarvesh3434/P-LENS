"""Adaptive mutation budget as a bounded resource account priced by a dual variable.

Selection at each step solves the constrained programme

    max_{M feasible}  E[V(M)] - lambda_t c(M) - mu churn(M) - nu risk(M)
    s.t.              c(M) <= B_t,  |M| <= m_max,
                      structural constraints (degree, connectivity, lambda*)

``B_t`` is the **bounded primal resource account**; ``lambda_t`` is the **dual
price** of one cost unit, updated by projected dual ascent in
:mod:`plens.economics`.  This is an explicit correction of the earlier draft,
which called the budget itself a dual variable.  The two are distinct:

* the account ``B_t`` is what the selector may spend;
* the price ``lambda_t`` is the marginal value of relaxing that account.

The account update is

    LCB_t   = mean_gain_t - z * std_gain_t
    safety  = 1 / (1 + a_o * overload_t + a_f * failure_t)
    damping = (1 - 0.5*u_t) * safety * ml_refill
    refill  = (b0 + beta * PD_t) * damping / (1 + d_p * lambda_t)
    credit  = (eta_g * LCB_t - eta_s * churn_t) * damping
    B_{t+1} = clip(B_t - spent_t + refill + credit
                   + kappa_mu*tanh(max(MU_t - MU_0, 0))*damping,
                   B_min, B_max)

with marginal utility ``MU_t = (V_hat_t - lambda_t c_t) / max(c_t, eps)``.
The price enters the mutation field as ``-lambda_t c_a``, so a candidate is
accepted only when its estimated value exceeds its dual-priced cost.  This gives
the primal-dual relationship the original debit/credit rule lacked.

``mode="legacy"`` reproduces the original linear update for comparison;
``mode="fixed"`` pins the budget to its initial value; ``enabled=False`` is the
``no_budget`` ablation and pins it to ``B_max``.
"""

from __future__ import annotations

from experiments.config import Config
from plens.economics import EconomyState, economy_step


def gate(budget: float, min_cost: float) -> float:
    return 1.0 if budget >= min_cost else 0.0


def update_budget(
    b: float,
    spent: float,
    mean_pd: float,
    mean_pos_gain: float,
    mean_stability_cost: float,
    cfg: Config,
    enabled: bool = True,
    gain_std: float = 0.0,
    overload: float = 0.0,
    failure_rate: float = 0.0,
    uncertainty: float = 0.0,
    refill_scale: float = 1.0,
    mode: str | None = None,
    dual_price: float | None = None,
    marginal_utility: float | None = None,
    iteration: int = 0,
    churn: float = 0.0,
) -> float:
    """Advance the resource account by one step and return the new budget.

    ``dual_price``/``marginal_utility`` are optional because the public helper
    is also used in unit tests and ablations; when omitted the price is taken
    from the configuration's initial value.
    """
    price = float(cfg.budget_price_initial if dual_price is None else dual_price)
    state = EconomyState(budget=float(b), price=price)
    out = economy_step(
        state,
        spent=spent,
        mean_pd=mean_pd,
        mean_gain=mean_pos_gain,
        mean_stability=mean_stability_cost,
        cfg=cfg,
        enabled=enabled,
        gain_std=gain_std,
        overload=overload,
        failure_rate=failure_rate,
        uncertainty=uncertainty,
        refill_scale=refill_scale,
        mode=mode,
        iteration=iteration,
        churn=churn,
    )
    return float(out.budget)
