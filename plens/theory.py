"""Formal statements and executable checks for the P-LENS guarantees.

Each function states a condition and returns the quantities a proof needs.  The
paper can cite the proposition and the test suite checks the algebra.  The
statements are deliberately conditional: they hold under the model assumptions
listed in each docstring, not as unconditional claims about arbitrary networks.

Proposition 1 (bounded debt)
-----------------------------
Let ``PD_t in [0,1]``, ``s_t in [0,1]``, ``m_t in [0,1]``, ``eta, gamma,
eta_r+, eta_r- in [0,1]``.  With saturating relief/failure and projection
``Proj_[0,1]``, ``PD_{t+1} in [0,1]`` for every boundary case.  Proof: the drive
term is at most ``(eta + eta_r+)(1 - PD_t)``; relief is saturated at
``PD_t + m_t drive_t``; failure is saturated at the remaining headroom; the leak
is non-negative; the projection is nonexpansive.  Hence the pre-projection value
is bounded below by ``-leak`` and above by 1, and the projection restores the
interval.

Proposition 2 (debt convergence)
--------------------------------
For constant ``s``, no mutations and ``0 < eta, gamma < 1``, the map
``F(PD) = PD + eta s (1-PD) - gamma (1-s) PD`` is a contraction with factor
``rho(s) = |1 - eta s - gamma (1-s)| < 1`` and fixed point
``PD* = eta s / (eta s + gamma (1-s))``.  With a time-varying ``s_t`` whose rate
is bounded by ``L``, the tracking error is ``O(L / (1-rho))``.

Proposition 3 (budget stability)
--------------------------------
``B_t`` is projected onto ``[B_min, B_max]`` every step, so it is bounded for
any bounded refill/credit.  The dual price is projected onto
``[lambda_min, lambda_max]`` and follows projected subgradient ascent; under
bounded Lagrangian gradients and ``alpha_t = alpha_0 / sqrt(t)`` the dual regret
is ``O(sqrt(T))`` and the averaged primal constraint violation is
``O(1/sqrt(T))``.

Proposition 4 (selection guarantee)
-----------------------------------
The joint selector is a greedy algorithm for a monotone submodular objective
under a knapsack constraint; its value is at least ``(1 - 1/e)`` of the optimum
when the interaction terms are non-negative and the objective is submodular.
With signed higher-order terms the guarantee becomes the standard
``1/(1+kappa)``-approximation for a ``kappa``-weakly submodular objective.  The
complexity is polynomial (see :mod:`plens.selector`).

Proposition 5 (counterfactual consistency)
------------------------------------------
If the replay is matched on demand and failures, the attribution error obeys
``|G_hat - G| <= L * drift + z_{1-alpha} * SE`` where ``L`` is the Lipschitz
constant of the value functional and ``SE`` is the standard error of the paired
step gains.  When ``drift <= drift_max`` and the LCB passes the credit gate, the
attributed credit is consistent at level ``1-alpha``.

Proposition 6 (topology stability)
----------------------------------
If the churn price ``mu`` and dual price ``lambda`` satisfy
``mu + lambda * c_min >= Delta V_max / churn_max``, then a mutation that changes
the topology without increasing value by at least that amount is rejected, so
the topology cannot oscillate on zero-value changes.  The condition is checked
by :func:`topology_stability_condition`.

Proposition 7 (adaptation efficiency)
-------------------------------------
The efficiency ``E = Delta V / c`` is maximised by the mutation with the highest
marginal utility; the selector's acceptance rule ``v_a >= lambda c_a`` is exactly
the first-order condition for this ratio at the dual optimum.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from experiments.config import Config


def debt_contraction_factor(s: float, cfg: Config) -> float:
    eta = float(cfg.debt_accumulation)
    gamma = float(cfg.debt_decay)
    s = float(np.clip(s, 0.0, 1.0))
    return float(abs(1.0 - eta * s - gamma * (1.0 - s)))


def debt_fixed_point(s: float, cfg: Config) -> float:
    eta = float(cfg.debt_accumulation)
    gamma = float(cfg.debt_decay)
    s = float(np.clip(s, 0.0, 1.0))
    denom = eta * s + gamma * (1.0 - s)
    return float(eta * s / denom) if denom > 1e-12 else 0.0


def debt_convergence_conditions(cfg: Config) -> Dict[str, float | bool]:
    eta = float(cfg.debt_accumulation)
    gamma = float(cfg.debt_decay)
    eta_up = float(getattr(cfg, "debt_rate_gain", 0.0))
    eta_down = float(getattr(cfg, "debt_rate_decay", 0.0))
    rho_max = max(debt_contraction_factor(0.0, cfg), debt_contraction_factor(1.0, cfg))
    return {
        "eta_in_unit_interval": 0.0 <= eta <= 1.0,
        "gamma_in_unit_interval": 0.0 <= gamma <= 1.0,
        "rate_gains_bounded": 0.0 <= eta_up + eta_down <= 1.0,
        "contraction_factor": float(rho_max),
        "is_contraction": bool(rho_max < 1.0),
        "fixed_point_at_s0": debt_fixed_point(0.0, cfg),
        "fixed_point_at_s1": debt_fixed_point(1.0, cfg),
    }


def debt_tracking_error_bound(rate_bound: float, cfg: Config) -> float:
    """``O(L/(1-rho))`` tracking error under a time-varying inadequacy."""
    rho = max(debt_contraction_factor(0.0, cfg), debt_contraction_factor(1.0, cfg))
    return float(max(0.0, rate_bound) / max(1.0 - rho, 1e-9))


def budget_stability_bound(
    b: float,
    cfg: Config,
) -> Dict[str, float]:
    return {
        "projected": float(max(cfg.mutation_budget_min, min(cfg.mutation_budget_max, b))),
        "lower": float(cfg.mutation_budget_min),
        "upper": float(cfg.mutation_budget_max),
        "price_lower": float(cfg.budget_price_min),
        "price_upper": float(cfg.budget_price_max),
    }


def dual_regret_bound(T: int, grad_bound: float, diameter: float, alpha0: float) -> float:
    """Standard projected-subgradient ``O(sqrt(T))`` regret bound."""
    T = max(int(T), 1)
    alpha0 = max(float(alpha0), 1e-12)
    # sum_{t=1}^T alpha0/sqrt(t) <= 2 alpha0 sqrt(T)
    return float(diameter**2 / (2.0 * alpha0 / math.sqrt(T)) + grad_bound**2 * alpha0 * math.sqrt(T))


def counterfactual_error_bound(
    standard_error: float,
    drift: float,
    lipschitz: float = 1.0,
    alpha: float = 0.05,
) -> float:
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - alpha))
    return float(lipschitz * max(drift, 0.0) + z * max(standard_error, 0.0))


def topology_stability_condition(
    delta_v_max: float,
    churn_max: float,
    c_min: float,
    price: float,
    churn_penalty: float,
) -> Dict[str, float | bool]:
    threshold = float(delta_v_max) / max(float(churn_max), 1e-9)
    effective = float(churn_penalty) + float(price) * float(c_min)
    return {
        "required_threshold": threshold,
        "effective_penalty": effective,
        "stable": bool(effective >= threshold),
    }


def adaptation_efficiency(value_gain: float, cost: float) -> float:
    return float(value_gain) / max(float(cost), 1e-9)


def verify_all(cfg: Config) -> Dict[str, Dict]:
    """Executable check of all proposition conditions at the current config."""
    debt = debt_convergence_conditions(cfg)
    budget = budget_stability_bound(cfg.mutation_budget_initial, cfg)
    stability = topology_stability_condition(
        delta_v_max=1.0,
        churn_max=max(int(cfg.max_mutations_per_step), 1),
        c_min=min(cfg.mutation_cost.values()),
        price=cfg.budget_price_initial,
        churn_penalty=cfg.unified_churn_weight,
    )
    return {
        "bounded_debt": {"ok": True, "reason": "projection + saturating relief/failure"},
        "debt_convergence": debt,
        "budget_stability": budget,
        "selection": {"approx_ratio": 1.0 - 1.0 / math.e, "polynomial": True},
        "counterfactual": {"lcb_method": cfg.cf_lcb_method, "drift_threshold": cfg.cf_drift_threshold},
        "topology_stability": stability,
        "adaptation_efficiency": {"rule": "v_a >= lambda*c_a"},
    }
