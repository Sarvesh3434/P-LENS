"""[ML] The stress forecast must influence control, not the workload.

Earlier revisions scaled the offered-demand target by the forecast.  That made
ML runs face a different (hotter) workload than no-ML runs, so the comparison
was confounded and the ablation could look better simply because it saw less
traffic.  The forecast now modulates only internal urgency: a lower debt
tolerance and a higher budget refill.  These tests pin that contract down.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.config import Config
from experiments.runner import run_simulation
from network.graph import EvolvingNetwork
from plens.controller import PLENSController


class _FixedStress:
    """Stub predictor: always returns the configured label. [ML]"""

    def __init__(self, label: str, L: int):
        self.label = label
        self.L = L

    def predict_stress(self, window_hist_scaled: np.ndarray) -> dict:
        return {
            "label": self.label,
            "class": {"Low": 0, "Medium": 1, "High": 2}[self.label],
            "attack_prob": {"Low": 0.05, "Medium": 0.50, "High": 0.95}[self.label],
            "uncertainty": 0.1,
        }


@pytest.fixture
def ml_cfg() -> Config:
    return Config(
        network_size=12,
        initial_topology="ring",
        simulation_steps=40,
        failure_rate_node=0.0,
        failure_rate_edge=0.0,
        window_length_L=6,
        max_mutations_per_step=4,
        seed=11,
    )


def _demand(cfg: Config) -> np.ndarray:
    rng = np.random.default_rng(42)
    d = rng.random((cfg.network_size, cfg.network_size)) * 0.1
    np.fill_diagonal(d, 0.0)
    d = d + d.T
    np.fill_diagonal(d, 0.0)
    return d


def _run(cfg: Config, demand, predictor) -> dict:
    series = np.stack([demand] * cfg.simulation_steps)
    scaled = np.random.default_rng(0).random((cfg.simulation_steps, cfg.window_length_L)) * 0.5
    return run_simulation(cfg, series, scaled_feat=scaled, predictor=predictor)


def test_forecast_does_not_change_offered_workload(ml_cfg):
    """High/Low forecasts must see exactly the same demand at every step."""
    demand = _demand(ml_cfg)
    hi = _run(ml_cfg, demand, _FixedStress("High", ml_cfg.window_length_L))
    lo = _run(ml_cfg, demand, _FixedStress("Low", ml_cfg.window_length_L))
    hi_dem = [x["metrics"]["total_demand"] for x in hi["history"]]
    lo_dem = [x["metrics"]["total_demand"] for x in lo["history"]]
    assert hi_dem == pytest.approx(lo_dem, rel=1e-12)


def test_forecast_changes_controller_urgency(ml_cfg):
    """Same traffic/failures; only the forecast label differs."""
    net_a = EvolvingNetwork(ml_cfg, np.random.default_rng(5))
    net_b = EvolvingNetwork(ml_cfg, np.random.default_rng(5))
    ctl_a = PLENSController(ml_cfg, net_a, ml_predictor=_FixedStress("High", ml_cfg.window_length_L))
    ctl_b = PLENSController(ml_cfg, net_b, ml_predictor=_FixedStress("Low", ml_cfg.window_length_L))
    d = _demand(ml_cfg)
    for t in range(20):
        hist = np.zeros((ml_cfg.window_length_L, 9))
        ctl_a.step(t, d, hist)
        ctl_b.step(t, d, hist)
    tol_a = ctl_a.history[-1]["ml"]["tolerance_factor"]
    tol_b = ctl_b.history[-1]["ml"]["tolerance_factor"]
    assert tol_a < tol_b, "attack forecast must tighten the debt tolerance"
    bud_a = [x["budget"] for x in ctl_a.history]
    bud_b = [x["budget"] for x in ctl_b.history]
    assert bud_a != bud_b, "forecast must change the budget trajectory"


def test_no_ml_matches_no_predictor(ml_cfg):
    """ablation='no_ml' must behave exactly like having no predictor at all."""
    demand = _demand(ml_cfg)
    cfg_noml = Config(**{**ml_cfg.__dict__, "ablation": "no_ml"})
    r1 = _run(cfg_noml, demand, _FixedStress("High", ml_cfg.window_length_L))
    r2 = _run(cfg_noml, demand, None)
    h1 = [(x["metrics"]["congestion"], x["metrics"]["latency"]) for x in r1["history"]]
    h2 = [(x["metrics"]["congestion"], x["metrics"]["latency"]) for x in r2["history"]]
    assert h1 == h2
