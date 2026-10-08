"""P-LENS pipeline entry point.

End-to-end demo: load data → (train ML if needed) → run experiment
(P-LENS vs baselines, CRN across seeds) → ablations → figures. All artifacts
land in results/ (JSON/CSVs are tracked, PNGs are gitignored).

Usage:
    python main.py                      # synthetic data, demo scale (~1 min)
    python main.py --dataset cic        # real CIC-IDS2017 windows (needs DataSets)
    python main.py --dataset unsw
    python main.py --steps 120 --seeds 5 --skip-ablations
    python main.py --fast               # tiny run for smoke-testing

Examples:
    python main.py --fast --dataset synthetic
    python main.py --dataset cic --seeds 3 --steps 60
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.config import Config, default_config
from experiments.runner import run_experiment
from experiments.ablations import run_ablations
from visualization.plots import make_all_figures


def build_demand_series(cfg: Config, bundle) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-step demand matrices + (optionally) scaled features for the sim.

    The sim consumes one demand matrix per step [SIM]; for real datasets we
    replay the 'sim' segment windows in order, looping if the run is longer
    than the segment. Synthetic data is generated for exactly `steps` windows.
    """
    steps = cfg.simulation_steps
    if cfg.dataset == "synthetic" or bundle is None:
        from ml.synthetic import synthetic_windows

        feats, dem, _ = synthetic_windows(cfg, n_windows=steps, seed=cfg.seed + 5_000)
        return dem, None
    lo, hi = bundle["splits"]["sim"]
    dem_all = bundle["demand"][lo:hi]
    scaled = bundle["scaled"]
    # loop the segment if the simulation needs more steps than we have windows
    reps = int(np.ceil(steps / max(len(dem_all), 1)))
    dem = np.concatenate([dem_all] * max(reps, 1), axis=0)[:steps]
    scaled_feat = scaled[lo : lo + len(dem_all)]
    scaled_feat = np.concatenate([scaled_feat] * max(reps, 1), axis=0)[:steps]
    return dem, scaled_feat


def main() -> None:
    ap = argparse.ArgumentParser(description="P-LENS end-to-end pipeline")
    ap.add_argument("--dataset", choices=("cic", "unsw", "synthetic"), default="synthetic")
    ap.add_argument("--steps", type=int, default=None, help="simulation steps")
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds")
    ap.add_argument("--fast", action="store_true", help="tiny smoke-test run")
    ap.add_argument("--skip-train", action="store_true", help="reuse existing model.pt")
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    overrides = {"dataset": args.dataset}
    if args.steps is not None:
        overrides["simulation_steps"] = args.steps
    if args.seeds is not None:
        overrides["n_seeds"] = args.seeds
    if args.fast:
        overrides.update(
            {
                "simulation_steps": 30,
                "n_seeds": 2,
                "epochs": 3,
                "synthetic_windows": 200,
                "max_rows_per_file": 8000,
                "flows_per_window": 200,
            }
        )
    cfg = default_config(**overrides)

    t0 = time.time()
    print(f"[1/4] dataset = {cfg.dataset}")

    # --- ML stage [ML] ---
    predictor = None
    bundle = None
    want_ml = cfg.dataset != "synthetic"
    model_pt = cfg.model_path()
    if want_ml and args.skip_train and model_pt.exists():
        from ml.predict import StressPredictor

        print(f"      reusing frozen model {model_pt}")
        predictor = StressPredictor(model_pt, cfg.scaler_path())
        # rebuild the window bundle for the sim segment without retraining
        from ml.train import prepare_bundle

        bundle = prepare_bundle(cfg)
    elif want_ml:
        from ml.predict import TrainedPredictor
        from ml.train import prepare_bundle, train_model

        print("      training GRU (early stopping on val macro-F1)")
        out = train_model(cfg)
        # wrap the in-memory model in the predictor interface the controller
        # expects (predict_stress on scaled (L, F) windows) [ML]
        predictor = TrainedPredictor(out["model"], out["bundle"]["scaler"], cfg.window_length_L)
        bundle = out["bundle"]

    # --- ML evaluation: test-set accuracy -> results/ml_metrics.json [ML] ---
    if want_ml and bundle is not None:
        from ml.evaluate import run_evaluation

        model_for_eval = getattr(predictor, "model", None)
        if model_for_eval is not None:
            metrics = run_evaluation(cfg, model_for_eval, bundle)
            print(
                f"      accuracy = {metrics['accuracy']:.4f}  "
                f"macro_f1 = {metrics['macro_f1']:.4f}  "
                f"(beats_persistence={metrics['beats_persistence']})"
            )
            print(f"      wrote {cfg.artifacts_dir() / 'ml_metrics.json'}")

    # --- demand series for the simulation [SIM] ---
    demand_series, scaled_feat = build_demand_series(cfg, bundle)
    print(f"[2/4] experiment: {cfg.n_seeds} seeds x 4 policies x {cfg.simulation_steps} steps")
    res = run_experiment(
        cfg,
        demand_series,
        scaled_feat,
        predictor,
        return_runs=True,
        save_runs=True,
    )

    # --- ablations ---
    abl_summary = None
    if not args.skip_ablations:
        print("[3/4] ablations (no_ml / no_debt / no_credit / no_budget)")
        abl_summary = run_ablations(cfg, demand_series, scaled_feat, predictor)
    else:
        print("[3/4] ablations skipped")

    # --- figures ---
    if not args.no_figures:
        print("[4/4] figures -> results/")
        made = make_all_figures(
            runs=res.get("runs"),
            experiment_summary=res,
            ablation_summary=abl_summary,
            cfg=cfg,
        )
        for p in made:
            print(f"      {p.name}")

    print(f"done in {time.time() - t0:.1f}s -- artifacts in {cfg.artifacts_dir()}")


if __name__ == "__main__":
    main()
