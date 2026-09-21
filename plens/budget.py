"""[DIAGRAM] Global mutation budget B_t. [ASSUMPTION] refill from debt and delayed gains."""

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
) -> float:
    if not enabled:
        return cfg.mutation_budget_max
    nxt = (
        b
        - spent
        + cfg.budget_refill * mean_pd
        + cfg.eta_g * mean_pos_gain
        - cfg.eta_s * mean_stability_cost
    )
    return float(max(cfg.mutation_budget_min, min(cfg.mutation_budget_max, nxt)))
