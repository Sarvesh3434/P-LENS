"""[ML][SIM] Synthetic traffic-window generator + visualization helpers."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from experiments.config import Config, default_config
from ml.data_loader import FEATURE_NAMES
from ml.synthetic import synthetic_windows
from visualization.plots import (
    ablation_bars,
    make_all_figures,
    metric_timeseries,
    policy_bars,
    topology_evolution,
    training_curves,
)


def test_synthetic_shapes_match_feature_names():
    cfg = default_config(network_size=10, synthetic_windows=40)
    feats, dem, metas = synthetic_windows(cfg, n_windows=40, seed=0)
    assert feats.shape == (40, len(FEATURE_NAMES))
    assert dem.shape == (40, 10, 10)
    assert len(metas) == 40
    assert np.isfinite(feats).all()
    # demand symmetric, zero diagonal [SIM]
    assert np.allclose(dem, np.transpose(dem, (0, 2, 1)))
    assert np.allclose(np.diagonal(dem, axis1=1, axis2=2), 0.0)


def test_synthetic_diurnal_changes_load():
    cfg = default_config(network_size=8, synthetic_windows=96, synthetic_diurnal=True)
    feats, _, _ = synthetic_windows(cfg, n_windows=96, seed=1)
    # flow_count column: night windows should be quieter than the day peak
    fc = feats[:, 0]
    assert fc.max() > 1.5 * fc.min()


def test_synthetic_deterministic_per_seed():
    cfg = default_config(network_size=8, synthetic_windows=20)
    f1, d1, _ = synthetic_windows(cfg, n_windows=20, seed=5)
    f2, d2, _ = synthetic_windows(cfg, n_windows=20, seed=5)
    assert np.array_equal(f1, f2)
    assert np.array_equal(d1, d2)


def test_prepare_bundle_synthetic():
    """dataset='synthetic' must work end-to-end through prepare_bundle. [ML]"""
    from ml.train import prepare_bundle

    cfg = default_config(
        dataset="synthetic",
        network_size=8,
        synthetic_windows=300,
        flows_per_window=100,
        window_length_L=6,
        epochs=1,
        seed=3,
    )
    bundle = prepare_bundle(cfg)
    assert bundle["scaled"].shape[0] == 300
    tr = bundle["seq"]["train"]
    assert tr[0].shape[1:] == (cfg.window_length_L, len(FEATURE_NAMES))


def _tiny_run(cfg: Config, seed: int = 7) -> dict:
    """Minimal run dict shaped like run_simulation output."""
    from experiments.runner import run_simulation

    from ml.synthetic import synthetic_windows as sw

    cfg2 = Config(**{**cfg.__dict__, "seed": seed})
    _, dem, _ = sw(cfg2, n_windows=cfg.simulation_steps, seed=seed)
    return run_simulation(cfg2, dem)


def test_metric_timeseries_and_topology(tmp_path):
    import networkx as nx

    cfg = default_config(
        network_size=10,
        simulation_steps=12,
        initial_topology="ring",
        synthetic_windows=12,
    )
    runs = {"plens": [_tiny_run(cfg, 7)], "static": [_tiny_run(cfg, 7)]}
    for r in runs["plens"] + runs["static"]:
        # topology evolution needs n_nodes; run_simulation sets it [SIM]
        assert "topology_snapshots" in r
    p1 = metric_timeseries(runs, cfg, "latency", tmp_path)
    assert p1.exists() and p1.stat().st_size > 0
    p2 = topology_evolution(runs["plens"][0], cfg, out_dir=tmp_path)
    assert p2.exists() and p2.stat().st_size > 0


def test_policy_and_ablation_bars(tmp_path):
    cfg = default_config(network_size=10)
    summary = {
        "aggregate": {
            "plens": {"latency_mean": {"mean": 0.01, "ci95": 0.001, "values": [0.01]}},
            "static": {"latency_mean": {"mean": 0.012, "ci95": 0.001, "values": [0.012]}},
        }
    }
    p = policy_bars(summary, cfg, keys=("latency_mean",), out_dir=tmp_path)
    assert p.exists()
    abl = {
        "none": {"loss_mean": {"mean": 0.1, "ci95": 0.01, "values": [0.1]}},
        "no_debt": {"loss_mean": {"mean": 0.12, "ci95": 0.01, "values": [0.12]}},
    }
    p2 = ablation_bars(abl, cfg, keys=("loss_mean",), out_dir=tmp_path)
    assert p2.exists()


def test_training_curves(tmp_path):
    import json

    meta = {"history": {"train_loss": [1.0, 0.8, 0.6], "val_f1": [0.3, 0.5, 0.6]}}
    mp = tmp_path / "train_meta.json"
    mp.write_text(json.dumps(meta))
    p = training_curves(mp, tmp_path)
    assert p.exists() and p.name == "training_curves.png"
