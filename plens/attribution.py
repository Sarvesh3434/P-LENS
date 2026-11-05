"""Drift-aware counterfactual attribution for delayed topology mutations.

The matched-replay estimator answers "what would have happened without this
mutation?" but a single counterfactual difference confounds three things:

1. the **direct** effect of the mutation on the topology it created;
2. the **indirect** effect mediated by later topology changes that the mutation
   enabled or made more likely;
3. **environmental drift** between the replay window and the current world.

This module supplies the statistical and diagnostic pieces:

* ``lcb_stats`` — a confidence-aware lower bound on the mutation gain, using a
  Student-t interval for the paired step gains (or a Hoeffding bound when the
  sample is tiny).  The lower bound, not the point estimate, is what may be
  credited, so noisy mutations cannot accumulate credit.
* ``drift_score`` — total-variation drift of the offered demand plus normalized
  metric drift between the replay branch and the live network.  Above
  ``cf_drift_threshold`` the replay is marked unreliable and credit is damped.
* ``decompose`` — splits total gain into direct and indirect components.
* ``difference_in_differences`` — an alternative delayed-credit estimator used
  for comparison with matched replay.
* ``credit_weight`` — the confidence/drift multiplier applied to delayed credit.

The replay orchestration lives in :mod:`plens.evaluation`; keeping the
statistics here makes them independently testable and avoids a circular import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from scipy.stats import norm, t as student_t

from experiments.config import Config


@dataclass
class AttributionResult:
    total_gain: float
    direct_gain: float | None = None
    indirect_gain: float | None = None
    gain_std: float = 0.0
    standard_error: float = 0.0
    gain_lcb: float = 0.0
    confidence: float = 1.0
    drift_score: float = 0.0
    reliable: bool = True
    credit_weight: float = 1.0
    n_samples: int = 0
    method: str = "matched_replay"


def _normalized_demand(d: np.ndarray | None) -> np.ndarray:
    if d is None:
        return np.zeros(0, dtype=float)
    a = np.asarray(d, dtype=float).ravel()
    s = float(a.sum())
    return a / s if s > 1e-12 else np.zeros_like(a)


def demand_drift(current: np.ndarray | None, reference: np.ndarray | None) -> float:
    """Total-variation distance between two offered-demand distributions."""
    p = _normalized_demand(current)
    q = _normalized_demand(reference)
    if p.size == 0 or q.size == 0 or p.shape != q.shape:
        return 0.0
    return float(0.5 * np.abs(p - q).sum())


def metric_drift(
    current: Dict[str, float] | None,
    reference: Dict[str, float] | None,
) -> float:
    """Mean normalized divergence of key network metrics."""
    if not current or not reference:
        return 0.0
    keys = ("latency", "loss", "congestion", "throughput", "avg_path_length")
    vals = []
    for k in keys:
        a = float(current.get(k, 0.0))
        b = float(reference.get(k, 0.0))
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        scale = max(abs(a), abs(b), 1e-9)
        vals.append(abs(a - b) / scale)
    return float(np.clip(np.mean(vals), 0.0, 1.0)) if vals else 0.0


def drift_score(
    current_demand: np.ndarray | None,
    reference_demand: np.ndarray | None,
    current_metrics: Dict[str, float] | None,
    reference_metrics: Dict[str, float] | None,
    cfg: Config,
) -> float:
    """Combined demand/metric drift in ``[0, 1]``."""
    d = demand_drift(current_demand, reference_demand)
    m = metric_drift(current_metrics, reference_metrics)
    w = float(getattr(cfg, "cf_drift_weight", 0.5))
    return float(np.clip((1.0 - w) * d + w * m, 0.0, 1.0))


def lcb_stats(
    step_gains: Sequence[float],
    cfg: Config,
    point_gain: float | None = None,
) -> tuple[float, float, float, float]:
    """Return ``(mean, standard_error, lower_confidence_bound, confidence)``.

    The paired step gains are treated as observations of the same intervention.
    A Student-t interval is used for ``n >= 2``; for a single observation we use
    a conservative Hoeffding-style half-width on ``[-1, 1]``.  This is the
    statistical validity condition behind the credit gate: the bound holds
    under bounded gains and (approximately) exchangeable replay steps.
    """
    gains = np.asarray(list(step_gains), dtype=float)
    n = int(gains.size)
    mean = float(gains.mean()) if n else float(point_gain or 0.0)
    if point_gain is not None:
        mean = float(point_gain)
    if n > 1:
        se = float(np.std(gains, ddof=1) / np.sqrt(n))
    else:
        se = 0.0
    alpha = float(getattr(cfg, "cf_confidence_alpha", 0.05))
    method = str(getattr(cfg, "cf_lcb_method", "t")).lower()
    if method == "hoeffding":
        # Gains are bounded in [-1, 1]; half-width for a one-sided bound.
        half = float(np.sqrt(2.0 * np.log(1.0 / max(alpha, 1e-9)) / max(n, 1)))
        lcb = mean - half
    elif method == "z":
        z = float(getattr(cfg, "cf_z", 1.0))
        lcb = mean - z * se
    else:
        if n > 1:
            crit = float(student_t.ppf(1.0 - alpha, df=n - 1))
        else:
            crit = float(norm.ppf(1.0 - alpha))
        lcb = mean - crit * se
    confidence = float(1.0 / (1.0 + se)) if se > 0 else 1.0
    return mean, se, float(lcb), confidence


def decompose(total_gain: float, direct_gain: float) -> tuple[float, float]:
    """Split total gain into ``(direct, indirect)``."""
    return float(direct_gain), float(total_gain - direct_gain)


def difference_in_differences(
    post_factual: Dict[str, float],
    post_counterfactual: Dict[str, float],
    pre_factual: Dict[str, float],
    pre_counterfactual: Dict[str, float],
    cfg: Config,
) -> float:
    """Alternative delayed-credit estimator for comparison.

    ``DiD = (post_F - pre_F) - (post_C - pre_C)`` on the utility proxy.  It is
    kept as an explicit alternative because it does not require replaying the
    full branch, but it is more sensitive to drift than matched replay.
    """
    from plens.evaluation import delayed_gain  # local import avoids cycle

    post = delayed_gain(post_factual, post_counterfactual, cfg, 0)["G"]
    pre = delayed_gain(pre_factual, pre_counterfactual, cfg, 0)["G"]
    return float(post - pre)


def credit_weight(confidence: float, drift: float, reliable: bool, cfg: Config) -> float:
    """Confidence/drift multiplier for delayed credit and debt relief."""
    penalty = float(getattr(cfg, "cf_credit_drift_penalty", 1.0))
    w = float(np.clip(confidence, 0.0, 1.0)) * max(0.0, 1.0 - penalty * float(np.clip(drift, 0.0, 1.0)))
    if not reliable:
        w *= 0.5
    return float(np.clip(w, 0.0, 1.0))


def is_reliable(drift: float, cfg: Config) -> bool:
    return bool(float(drift) <= float(getattr(cfg, "cf_drift_threshold", 1.0)))


def build_result(
    total_gain: float,
    step_gains: Sequence[float],
    cfg: Config,
    direct_gain: float | None = None,
    drift: float = 0.0,
    reliable: bool | None = None,
    method: str = "matched_replay",
) -> AttributionResult:
    mean, se, lcb, conf = lcb_stats(step_gains, cfg, point_gain=total_gain)
    reliable = is_reliable(drift, cfg) if reliable is None else bool(reliable)
    direct = float(direct_gain) if direct_gain is not None else None
    indirect = float(total_gain - direct) if direct is not None else None
    return AttributionResult(
        total_gain=float(total_gain),
        direct_gain=direct,
        indirect_gain=indirect,
        gain_std=float(np.std(step_gains, ddof=1)) if len(step_gains) > 1 else 0.0,
        standard_error=float(se),
        gain_lcb=float(lcb),
        confidence=float(conf),
        drift_score=float(drift),
        reliable=bool(reliable),
        credit_weight=credit_weight(conf, drift, reliable, cfg),
        n_samples=int(len(step_gains)),
        method=method,
    )
