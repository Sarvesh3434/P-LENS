"""Run the reviewer-requested scenario study.

Examples:
    python run_study.py --quick
    python run_study.py --scales 24 --topologies ring small_world \
        --patterns stationary --failures random --seeds 30 --steps 120
    python run_study.py --dataset cic --seeds 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiments.config import default_config
from experiments.scenarios import run_study


def _csv_list(value, cast=str):
    return [cast(x) for x in value.replace(",", " ").split() if x]


def main() -> None:
    ap = argparse.ArgumentParser(description="P-LENS scenario study")
    ap.add_argument("--dataset", choices=("cic", "unsw", "synthetic"), default="synthetic")
    ap.add_argument("--scales", default="24,50,100,200")
    ap.add_argument("--topologies", default="ring,erdos_renyi,barabasi_albert,small_world")
    ap.add_argument("--patterns", default="stationary,shift,periodic,bursty")
    ap.add_argument("--failures", default="random,targeted,multi,recovery")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--save-runs", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    args = ap.parse_args()

    if args.quick:
        args.scales = "24"
        args.topologies = "ring,small_world"
        args.patterns = "stationary,bursty"
        args.failures = "random,targeted"
        args.seeds = min(args.seeds, 5)
        args.steps = min(args.steps, 60)

    cfg = default_config(
        dataset=args.dataset,
        simulation_steps=args.steps,
        n_seeds=args.seeds,
        cf_bootstrap=args.bootstrap,
    )
    bundle = None
    predictor = None
    if args.dataset != "synthetic":
        from ml.predict import TrainedPredictor
        from ml.train import prepare_bundle, train_model

        print("[study] preparing real-data bundle and GRU")
        out = train_model(cfg)
        bundle = out["bundle"]
        predictor = TrainedPredictor(
            out["model"],
            out["bundle"]["scaler"],
            cfg.window_length_L,
            cfg.ml_horizon,
            threshold=out.get("threshold", 0.5),
        )

    t0 = time.time()
    summary = run_study(
        cfg,
        bundle=bundle,
        predictor=predictor,
        scales=_csv_list(args.scales, int),
        topologies=_csv_list(args.topologies),
        patterns=_csv_list(args.patterns),
        failures=_csv_list(args.failures),
        seeds=list(range(7, 7 + args.seeds)),
        out_dir=Path(args.out) if args.out else None,
        save_runs=args.save_runs,
        n_jobs=args.jobs,
    )
    print(f"[study] {len(summary['cells'])} cells in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
