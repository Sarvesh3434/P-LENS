"""[DIAGRAM] Figures for the P-LENS paper/demo. All functions save a PNG into
`cfg.artifacts_dir()` (results/) and return the path. PNGs are gitignored; the
JSON they were built from is not.

Conventions [ASSUMPTION]:
- metric time series are plotted per policy as mean across seeds (CI bands when
  ≥2 seeds are available);
- policy colors are stable across every figure for readability;
- topology evolution is drawn with fixed node layout (initial positions) so
  additions/prunes/rewires read as motion, not as layout reshuffles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: no display needed [ASSUMPTION]
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from experiments.config import Config

# stable policy colors across figures [ASSUMPTION]
POLICY_COLORS = {
    "plens": "#1f77b4",
    "static": "#7f7f7f",
    "random": "#ff7f0e",
    "greedy": "#2ca02c",
}
POLICY_LABELS = {
    "plens": "P-LENS",
    "static": "Static",
    "random": "Random-adaptive",
    "greedy": "Greedy",
}
METRIC_LABELS = {
    "latency": "Latency (s, delivered-weighted)",
    "throughput": "Throughput (delivered share)",
    "loss": "Loss fraction",
    "congestion": "Mean link utilization",
}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def metric_timeseries(
    runs: Dict[str, List[dict]],
    cfg: Config,
    metric: str = "latency",
    out_dir: str | Path | None = None,
) -> Path:
    """Per-step mean±95%CI of one metric per policy.

    runs: policy -> list of run dicts from run_simulation (each has 'history').
    """
    out_dir = Path(out_dir) if out_dir else cfg.artifacts_dir()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for policy, run_list in runs.items():
        series = []
        for run in run_list:
            series.append([x["metrics"].get(metric, np.nan) for x in run["history"]])
        T = min(len(s) for s in series)
        arr = np.array([s[:T] for s in series], dtype=float)
        mu = np.nanmean(arr, axis=0)
        ax.plot(mu, label=POLICY_LABELS.get(policy, policy), color=POLICY_COLORS.get(policy))
        if arr.shape[0] > 1:
            sd = np.nanstd(arr, axis=0, ddof=1)
            ax.fill_between(
                np.arange(T),
                mu - 1.96 * sd / np.sqrt(arr.shape[0]),
                mu + 1.96 * sd / np.sqrt(arr.shape[0]),
                color=POLICY_COLORS.get(policy),
                alpha=0.15,
                linewidth=0,
            )
    ax.set_xlabel("Simulation step t")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} over time")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, f"timeseries_{metric}.png")


def policy_bars(
    experiment_summary: dict,
    cfg: Config,
    keys: Sequence[str] = ("latency_mean", "throughput_mean", "loss_mean", "congestion_mean"),
    out_dir: str | Path | None = None,
) -> Path:
    """Bar chart: mean±95%CI per policy for the four headline metrics."""
    out_dir = Path(out_dir) if out_dir else cfg.artifacts_dir()
    agg = experiment_summary["aggregate"]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.6))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        policies = list(agg.keys())
        mus = [agg[p][key]["mean"] for p in policies]
        cis = [agg[p][key]["ci95"] for p in policies]
        colors = [POLICY_COLORS.get(p, "#999999") for p in policies]
        ax.bar(
            [POLICY_LABELS.get(p, p) for p in policies],
            mus,
            yerr=cis,
            color=colors,
            capsize=3,
        )
        base = key.removesuffix("_mean")
        ax.set_title(METRIC_LABELS.get(base, key), fontsize=9)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3, axis="y")
    return _save(fig, out_dir, "policy_bars.png")


def topology_evolution(
    run: dict,
    cfg: Config,
    steps: Sequence[int] | None = None,
    out_dir: str | Path | None = None,
) -> Path:
    """Draw G_t at a few snapshots with one fixed layout.

    steps defaults to [start, 1/3, 2/3, end]. Red dashed = removed vs previous,
    green solid = added vs previous. Node positions are fixed from the initial
    snapshot so changes read as motion [ASSUMPTION].
    """
    out_dir = Path(out_dir) if out_dir else cfg.artifacts_dir()
    snaps = run["topology_snapshots"]
    T = len(snaps)
    if steps is None:
        steps = sorted({0, T // 3, (2 * T) // 3, T - 1})
    n = run.get("n_nodes", cfg.network_size)
    G0 = nx.Graph()
    G0.add_nodes_from(range(n))
    G0.add_edges_from(map(tuple, snaps[0]))
    pos = nx.spring_layout(G0, seed=cfg.seed)  # deterministic layout [SIM]
    fig, axes = plt.subplots(1, len(steps), figsize=(4.0 * len(steps), 4.0))
    if len(steps) == 1:
        axes = [axes]
    prev: set = set()
    for ax, s in zip(axes, steps):
        cur = {tuple(sorted(e)) for e in snaps[s]}
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(cur)
        added = cur - prev
        removed = prev - cur
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60, node_color="#4472c4")
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(cur - added), width=1.0, alpha=0.5)
        if added:
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=list(added), width=1.8, edge_color="#2ca02c"
            )
        if removed:
            # removed edges are not in G; draw them from the full node set
            H = nx.Graph()
            H.add_nodes_from(range(n))
            H.add_edges_from(removed)
            nx.draw_networkx_edges(
                H, pos, ax=ax, edgelist=list(removed), width=1.8, edge_color="#d62728", style="dashed"
            )
        ax.set_title(f"t = {s}", fontsize=10)
        ax.set_axis_off()
        prev = cur
    fig.suptitle("Topology evolution (green = added, dashed red = removed vs previous)", fontsize=10)
    return _save(fig, out_dir, "topology_evolution.png")


def ablation_bars(
    ablation_summary: dict,
    cfg: Config,
    keys: Sequence[str] = ("latency_mean", "loss_mean", "congestion_mean"),
    out_dir: str | Path | None = None,
) -> Path:
    """Bar chart of P-LENS ablations (none / no_ml / no_debt / ...)."""
    out_dir = Path(out_dir) if out_dir else cfg.artifacts_dir()
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.6))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        abls = list(ablation_summary.keys())
        mus = [ablation_summary[a][key]["mean"] for a in abls]
        cis = [ablation_summary[a][key]["ci95"] for a in abls]
        ax.bar(abls, mus, yerr=cis, color="#1f77b4", capsize=3)
        base = key.removesuffix("_mean")
        ax.set_title(METRIC_LABELS.get(base, key), fontsize=9)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3, axis="y")
    return _save(fig, out_dir, "ablation_bars.png")


def training_curves(
    train_meta_path: str | Path,
    out_dir: str | Path | None = None,
    cfg: Config | None = None,
) -> Path:
    """Loss and validation macro-F1 per epoch from train_meta.json."""
    meta = json.loads(Path(train_meta_path).read_text())
    hist = meta["history"]
    out_dir = Path(out_dir) if (out_dir or cfg is None) else cfg.artifacts_dir()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.5))
    ax1.plot(hist["train_loss"], color="#1f77b4")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax1.set_title("GRU training loss")
    ax1.grid(alpha=0.3)
    ax2.plot(hist["val_f1"], color="#2ca02c")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation macro-F1")
    ax2.set_title("Validation macro-F1")
    ax2.grid(alpha=0.3)
    return _save(fig, out_dir, "training_curves.png")


def make_all_figures(
    runs: Optional[Dict[str, List[dict]]] = None,
    experiment_summary: Optional[dict] = None,
    ablation_summary: Optional[dict] = None,
    cfg: Config = None,
    out_dir: str | Path | None = None,
) -> List[Path]:
    """One-call figure generation for main.py. Skips inputs it wasn't given."""
    cfg = cfg if cfg is not None else Config()
    out_dir = Path(out_dir) if out_dir else cfg.artifacts_dir()
    made: List[Path] = []
    if runs:
        for metric in ("latency", "throughput", "loss", "congestion"):
            made.append(metric_timeseries(runs, cfg, metric, out_dir))
        first_policy = next(iter(runs))
        if runs[first_policy]:
            made.append(topology_evolution(runs[first_policy][0], cfg, out_dir=out_dir))
    if experiment_summary:
        made.append(policy_bars(experiment_summary, cfg, out_dir=out_dir))
    if ablation_summary:
        made.append(ablation_bars(ablation_summary, cfg, out_dir=out_dir))
    tm = out_dir / "train_meta.json"
    if tm.exists():
        made.append(training_curves(tm, out_dir, cfg))
    return made
