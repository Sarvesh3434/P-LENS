"""Adaptive mutation budget as the dual variable of a constrained programme.

Selection at each step is the constrained problem

    max_{M feasible}  E[V(M)] - lambda_c churn(M) - lambda_r risk(M)
    s.t.              cost(M) <= B_t,  |M| <= m_max,
                      structural constraints (degree, connectivity, lambda*)

where ``V`` combines latency, loss, throughput and resilience.  The budget
``B_t`` is the scalar price of structural change.  Its update is a bounded
debit/credit account driven by debt and by the lower confidence bound of the
measured mutation gain:

    LCB_t   = mean_gain_t - z * std_gain_t
    safety  = 1 / (1 + a_o * overload_t + a_f * failure_t)
    refill  = (b0 + beta * PD_t) * (1 - 0.5*u_t) * safety * ml_refill
    credit  = (eta_g * LCB_t - eta_s * stability_t) * (1 - 0.5*u_t) * safety
    B_{t+1} = clip(B_t - spent_t + refill + credit, B_min, B_max)

Under persistent overload or repeated failure the refill is halved, so churn
decays unless the measured lower-confidence gain keeps justifying it.  The LCB
makes the controller conservative when the mutation-value estimate is noisy.
``B`` is bounded by construction (projection), and the selector only applies
feasible operations, so hard topology constraints are preserved regardless of
``B``.

``mode="legacy"`` reproduces the original linear update for comparison;
``mode="fixed"`` pins the budget to its initial value; ``enabled=False`` is the
``no_budget`` ablation and pins it to ``B_max``.
"""

from __future__ import annotations

from experiments.config import Config


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
) -> float:
    if not enabled:
        return cfg.mutation_budget_max
    mode = mode or cfg.budget_mode
    if mode == "fixed":
        return float(cfg.mutation_budget_initial)
    if mode == "legacy":
        nxt = (
            b
            - spent
            + cfg.budget_refill * mean_pd
            + cfg.eta_g * mean_pos_gain
            - cfg.eta_s * mean_stability_cost
        )
        return float(max(cfg.mutation_budget_min, min(cfg.mutation_budget_max, nxt)))

    lcb = float(mean_pos_gain) - cfg.budget_uncertainty_z * max(float(gain_std), 0.0)
    safety = 1.0 / (
        1.0
        + cfg.budget_overload_damping * max(float(overload), 0.0)
        + cfg.budget_failure_damping * max(float(failure_rate), 0.0)
    )
    damping = (1.0 - 0.5 * float(np_clip(uncertainty))) * max(float(refill_scale), 0.0) * safety
    refill = (cfg.budget_base_refill + cfg.budget_refill * max(float(mean_pd), 0.0)) * damping
    # The measured (lower-confidence) mutation gain and its stability price are
    # a signed credit line: evidence that mutations pay raises the account even
    # when current debt is low, while churn without measurable benefit lowers it.
    credit = (cfg.eta_g * lcb - cfg.eta_s * max(float(mean_stability_cost), 0.0)) * damping
    # Persistent overload / repeated failure dampens (but does not freeze) the
    # account: spending still costs, so churn decays when actions stop helping.
    if overload >= cfg.budget_overload_threshold or failure_rate >= 0.5:
        refill *= 0.5
    nxt = b - spent + refill + credit
    return float(max(cfg.mutation_budget_min, min(cfg.mutation_budget_max, nxt)))


def np_clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))

