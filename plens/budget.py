"""Adaptive mutation budget as the dual variable of a constrained programme.

Selection at each step is the constrained problem

    max_{M feasible}  E[V(M)] - lambda_c churn(M) - lambda_r risk(M)
    s.t.              cost(M) <= B_t,  |M| <= m_max,
                      structural constraints (degree, connectivity, lambda*)

where ``V`` combines latency, loss, throughput and resilience.  The budget
``B_t`` is the scalar price of structural change.  Its update is a projected
dual-ascent / tracking rule:

    LCB_t   = mean_gain_t - z * std_gain_t / sqrt(n_t)
    eff_t   = (LCB_t - eta_s * mean_stability_t) / max(spent_t, 1)
    desired = B_min + (B_max-B_min) * (1 + tanh(kappa*(eff_t - rho))) / 2
    safety  = 1 / (1 + a_o * overload_t + a_f * failure_t)
    B_{t+1} = clip(B_t - spent_t + beta * PD_t * (1-u_t)
                   + alpha * (desired * safety - B_t), B_min, B_max)

with an additional freeze when the network has been persistently overloaded or
mutations have repeatedly failed.  The use of the lower confidence bound makes
the controller conservative when the mutation-value estimate is noisy; the
safety factor prevents "thrashing" during repeated failures.  ``B`` is bounded
by construction (projection), and the selector only applies feasible
operations, so hard topology constraints are preserved regardless of ``B``.

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
    efficiency = (lcb - cfg.eta_s * float(mean_stability_cost)) / max(float(spent), 1e-9)
    shaped = 0.5 * (1.0 + _tanh(2.0 * (efficiency - cfg.budget_marginal_target)))
    desired = cfg.mutation_budget_min + (
        cfg.mutation_budget_max - cfg.mutation_budget_min
    ) * shaped
    safety = 1.0 / (
        1.0
        + cfg.budget_overload_damping * max(float(overload), 0.0)
        + cfg.budget_failure_damping * max(float(failure_rate), 0.0)
    )
    desired *= safety
    refill = (cfg.budget_base_refill + cfg.budget_refill * max(float(mean_pd), 0.0)) * (1.0 - 0.5 * float(np_clip(uncertainty)))
    nxt = (
        b
        - spent
        + refill
        + cfg.budget_adapt * (desired - b)
    )
    # Persistent overload / repeated failure: do not let the account grow.
    if overload >= cfg.budget_overload_threshold or failure_rate >= 0.5:
        nxt = min(nxt, b)
    return float(max(cfg.mutation_budget_min, min(cfg.mutation_budget_max, nxt)))


def _tanh(x: float) -> float:
    # avoid importing numpy in this tiny leaf module
    import math

    return math.tanh(x)


def np_clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))

