"""[DIAGRAM]+[ASSUMPTION] Ablation study: P-LENS with one component disabled.

Runs the same seeds/traffic/failures as the main experiment (common random
numbers) with `ablation` set to one of:
    none            – full model
    no_ml           – prediction detached from load target and budget
    no_debt         – PD forced to 0 (debt term drops out of MF, refill loses mean(PD))
    no_credit       – MC stays empty (credit term drops out of MF)
    no_budget       – budget pinned to B_max (no throttle; still capped per step)
    no_uncertainty  – point-gain credit/relief instead of the lower confidence bound
    no_interaction  – marginal (score-only) selection instead of joint scoring
    fixed_score     – uniform mutation score (simple heuristic ranking)
Each ablation runs the full P-LENS machinery — none is replaced by another
policy — so the comparison isolates exactly one mechanism.  ``no_budget``
combined with ``no_debt``/``no_credit`` etc. can be requested as a comma-
separated string for interaction ablations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.config import Config
from experiments.runner import run_experiment


def run_ablations(
    cfg: Config,
    demand_series: np.ndarray,
    scaled_feat: np.ndarray | None = None,
    predictor=None,
    seeds=None,
    out_dir: str | Path | None = None,
    save_runs: bool = False,
    verbose: bool = True,
    n_jobs: int = 1,
    ablations: list[str] | None = None,
) -> dict:
    seeds = seeds if seeds is not None else list(range(7, 7 + cfg.n_seeds))
    out_dir = Path(out_dir) if out_dir else cfg.artifacts_dir() / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    if ablations is None:
        ablations = [
            "none",
            "no_ml",
            "no_debt",
            "no_credit",
            "no_budget",
            "no_uncertainty",
            "no_interaction",
            "fixed_score",
        ]
    all_rows: dict = {}
    for abl in ablations:
        if verbose:
            print(f"[ablation] {abl}")
        run_cfg = Config(**{**cfg.__dict__, "ablation": abl, "policy": "plens"})
        res = run_experiment(
            run_cfg,
            demand_series,
            scaled_feat,
            predictor,
            policies=("plens",),
            seeds=seeds,
            out_dir=out_dir / abl,
            save_runs=save_runs,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        all_rows[abl] = res["aggregate"]["plens"]
    with open(out_dir / "ablation_summary.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    return all_rows
