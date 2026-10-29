"""Paired statistical analysis for the P-LENS study.

The study uses common random numbers (same seed = same topology, traffic and
failure draws), so every baseline is paired with P-LENS at the seed level.  We
report:

* mean difference and its bootstrap 95% CI;
* paired Cohen's d_z (effect size);
* two-sided paired t-test and Wilcoxon signed-rank test;
* Holm-Bonferroni adjusted p-values across the metric family;
* achieved power at the observed effect size and seed count.

The normal-approximation power is used because ``statsmodels`` is not a project
dependency; it is accurate for the n >= 30 study and conservative for smaller
pilots.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from scipy.stats import norm, ttest_rel, wilcoxon


def bootstrap_ci(values: Iterable[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    a = np.asarray(list(values), dtype=float)
    if len(a) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(a, size=len(a), replace=True)) for _ in range(n_boot)])
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def cohens_dz(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    d = x - y
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    return float(np.mean(d) / sd) if sd > 1e-12 else 0.0


def achieved_power(dz: float, n: int, alpha: float = 0.05) -> float:
    """Two-sided paired-test power under a normal approximation."""
    if n <= 1:
        return 0.0
    ncp = abs(dz) * np.sqrt(n)
    crit = norm.ppf(1.0 - alpha / 2.0)
    return float(1.0 - norm.cdf(crit - ncp) + norm.cdf(-crit - ncp))


def holm_bonferroni(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down adjustment over a metric family."""
    items = [(k, (1.0 if v is None or not np.isfinite(v) else float(v))) for k, v in pvalues.items()]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, float] = {}
    running = 0.0
    for rank, (k, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[k] = float(running)
    return out


def paired_tests(a: Iterable[float], b: Iterable[float], metric: str = "") -> dict:
    """Paired comparison of P-LENS (a) against a baseline (b)."""
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    d = x - y
    n = len(d)
    res = {
        "metric": metric,
        "n": n,
        "mean_plens": float(np.mean(x)) if n else None,
        "mean_baseline": float(np.mean(y)) if n else None,
        "mean_diff": float(np.mean(d)) if n else None,
        "ci95_diff": bootstrap_ci(d, seed=17),
        "cohens_dz": cohens_dz(x, y),
        "power": achieved_power(cohens_dz(x, y), n),
        "ttest_p": None,
        "wilcoxon_p": None,
        "note": "",
    }
    if n < 2 or np.allclose(d, 0.0):
        res["note"] = "insufficient variation"
        return res
    try:
        res["ttest_p"] = float(ttest_rel(x, y).pvalue)
    except Exception as exc:  # pragma: no cover
        res["note"] = str(exc)
    try:
        res["wilcoxon_p"] = float(wilcoxon(x, y, zero_method="wilcox").pvalue)
    except Exception as exc:  # pragma: no cover
        res["note"] = str(exc)
    return res


def holm_family(results: Dict[str, dict], key: str = "wilcoxon_p") -> Dict[str, float]:
    pvals = {k: v.get(key) for k, v in results.items()}
    return holm_bonferroni({k: (1.0 if v is None else v) for k, v in pvals.items()})
