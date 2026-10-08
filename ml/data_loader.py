"""[ML] Load CIC-IDS2017 MachineLearningCVE and UNSW-NB15 shuffled splits.

[ASSUMPTION] These files have no Timestamp and no Source IP. Time is file order
then row order. Endpoints are hashed from Destination Port (CIC) or proto/service (UNSW).
The dataset supplies traffic/condition information only — not topology.
Attack labels are NEVER used as model inputs.
"""

from __future__ import annotations

import csv
import zlib
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

from experiments.config import Config


FEATURE_NAMES = [
    "flow_count",
    "pkt_rate",
    "byte_rate",
    "mean_duration",
    "mean_fwd_pkts",
    "mean_bwd_pkts",
    "mean_fwd_bytes",
    "mean_bwd_bytes",
    "unique_ports_frac",
    "attack_flag",  # 1 if any flow in window is an attack, else 0  [ML]
]


def _crc_node(text: str, n: int, salt: str = "plens") -> int:
    return zlib.crc32((text + salt).encode("utf-8")) % n


def _finite(x, default=0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return v


def _strip(h: str) -> str:
    return h.strip().lstrip("\ufeff")


def iter_cic_rows(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = [_strip(h) for h in next(reader)]
        idx = {name: i for i, name in enumerate(header)}

        def col(*names, default=0.0):
            for n in names:
                if n in idx:
                    return idx[n]
            return None

        i_port = col("Destination Port")
        i_dur = col("Flow Duration")
        i_fp = col("Total Fwd Packets")
        i_bp = col("Total Backward Packets")
        i_fb = col("Total Length of Fwd Packets")
        i_bb = col("Total Length of Bwd Packets")
        i_lab = col("Label")
        for row in reader:
            if not row or len(row) < 5:
                continue
            yield {
                "dst_port": int(_finite(row[i_port] if i_port is not None else 0)),
                "duration": max(_finite(row[i_dur] if i_dur is not None else 0), 0.0),
                "fwd_pkts": max(_finite(row[i_fp] if i_fp is not None else 0), 0.0),
                "bwd_pkts": max(_finite(row[i_bp] if i_bp is not None else 0), 0.0),
                "fwd_bytes": max(_finite(row[i_fb] if i_fb is not None else 0), 0.0),
                "bwd_bytes": max(_finite(row[i_bb] if i_bb is not None else 0), 0.0),
                "label": (row[i_lab] if i_lab is not None and i_lab < len(row) else "").strip(),
            }


def iter_unsw_rows(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {
                "proto": (row.get("proto") or "").strip(),
                "service": (row.get("service") or "").strip(),
                "duration": max(_finite(row.get("dur")), 0.0),
                "fwd_pkts": max(_finite(row.get("spkts")), 0.0),
                "bwd_pkts": max(_finite(row.get("dpkts")), 0.0),
                "fwd_bytes": max(_finite(row.get("sbytes")), 0.0),
                "bwd_bytes": max(_finite(row.get("dbytes")), 0.0),
                "rate": max(_finite(row.get("rate")), 0.0),
                "label": (row.get("attack_cat") or row.get("label") or "").strip(),
            }


def _flush_window(buf: List[dict], n_nodes: int, kind: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    n = len(buf)
    pkts = sum(r["fwd_pkts"] + r["bwd_pkts"] for r in buf)
    byts = sum(r["fwd_bytes"] + r["bwd_bytes"] for r in buf)
    dur = sum(r["duration"] for r in buf) / max(n, 1)
    feat = np.array(
        [
            float(n),
            pkts,
            byts,
            dur,
            sum(r["fwd_pkts"] for r in buf) / max(n, 1),
            sum(r["bwd_pkts"] for r in buf) / max(n, 1),
            sum(r["fwd_bytes"] for r in buf) / max(n, 1),
            sum(r["bwd_bytes"] for r in buf) / max(n, 1),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    demand = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    ports = set()
    attack_frac = 0.0
    for r in buf:
        if kind == "cic":
            ports.add(r["dst_port"])
            src = _crc_node(f"src:{r['dst_port']}", n_nodes, "src")
            dst = _crc_node(f"dst:{r['dst_port']}", n_nodes, "dst")
            lab = r["label"].upper()
            if lab and lab != "BENIGN":
                attack_frac += 1.0
        else:
            src = _crc_node(r.get("proto", "") + r.get("service", ""), n_nodes, "src")
            dst = _crc_node(r.get("service", "") + r.get("proto", "")[::-1], n_nodes, "dst")
            lab = r["label"]
            if lab and lab not in ("0", "Normal", "normal"):
                attack_frac += 1.0
        if src == dst:
            dst = (dst + 1) % n_nodes
        demand[src, dst] += r["fwd_bytes"] + r["bwd_bytes"]
    feat[8] = len(ports) / max(n, 1) if kind == "cic" else 0.0
    feat[9] = 1.0 if attack_frac > 0 else 0.0
    meta = {"attack_frac": attack_frac / max(n, 1), "n_flows": n}
    return feat, demand, meta


def windows_from_rows(rows: Iterator[dict], cfg: Config, kind: str):
    buf: List[dict] = []
    feats, demands, metas = [], [], []
    wsize = cfg.flows_per_window
    n_rows = 0
    for r in rows:
        buf.append(r)
        n_rows += 1
        if len(buf) >= wsize:
            f, d, m = _flush_window(buf, cfg.network_size, kind)
            feats.append(f)
            demands.append(d)
            metas.append(m)
            buf = []
        if cfg.max_rows_per_file and n_rows >= cfg.max_rows_per_file:
            break
    if buf:
        f, d, m = _flush_window(buf, cfg.network_size, kind)
        feats.append(f)
        demands.append(d)
        metas.append(m)
    return (
        np.stack(feats) if feats else np.zeros((0, len(FEATURE_NAMES))),
        np.stack(demands) if demands else np.zeros((0, cfg.network_size, cfg.network_size)),
        metas,
    )


def load_cic_windows(cfg: Config):
    feats_all = []
    dem_all = []
    meta_all = []
    file_bounds = []  # (name, start, end)
    start = 0
    for name in cfg.cic_files:
        path = cfg.cic_dir() / name
        if not path.exists():
            raise FileNotFoundError(path)
        f, d, m = windows_from_rows(iter_cic_rows(path), cfg, "cic")
        feats_all.append(f)
        dem_all.append(d)
        meta_all.extend(m)
        end = start + len(f)
        file_bounds.append((name, start, end))
        start = end
    feats = np.concatenate(feats_all, axis=0) if feats_all else np.zeros((0, len(FEATURE_NAMES)))
    dem = np.concatenate(dem_all, axis=0)
    return feats, dem, meta_all, file_bounds


def load_unsw_windows(cfg: Config):
    train_p = cfg.unsw_dir() / "UNSW_NB15_training-set.csv"
    test_p = cfg.unsw_dir() / "UNSW_NB15_testing-set.csv"
    f1, d1, m1 = windows_from_rows(iter_unsw_rows(train_p), cfg, "unsw")
    f2, d2, m2 = windows_from_rows(iter_unsw_rows(test_p), cfg, "unsw")
    feats = np.concatenate([f1, f2], axis=0)
    dem = np.concatenate([d1, d2], axis=0)
    metas = m1 + m2
    bounds = [("UNSW_train", 0, len(f1)), ("UNSW_test", len(f1), len(f1) + len(f2))]
    return feats, dem, metas, bounds


def chronological_split(n: int, bounds=None):
    """Four disjoint segments with 1-window gaps: train, val, test, sim. [ML]"""
    # Prefer file-based split for CIC (Mon/Tue train, Wed val, Thu test, Fri sim)
    if bounds and len(bounds) >= 8:
        # indices 0-1 Mon Tue, 2 Wed, 3-4 Thu, 5-7 Fri
        tr0, tr1 = bounds[0][1], bounds[1][2]
        va0, va1 = bounds[2][1], bounds[2][2]
        te0, te1 = bounds[3][1], bounds[4][2]
        sm0, sm1 = bounds[5][1], bounds[7][2]
        # gaps: drop last window of each preceding segment
        return {
            "train": (tr0, max(tr0, tr1 - 1)),
            "val": (va0, max(va0, va1 - 1)),
            "test": (te0, max(te0, te1 - 1)),
            "sim": (sm0, sm1),
        }
    a = int(0.45 * n)
    b = int(0.60 * n)
    c = int(0.78 * n)
    return {
        "train": (0, max(1, a - 1)),
        "val": (a + 1, max(a + 2, b - 1)),
        "test": (b + 1, max(b + 2, c - 1)),
        "sim": (c + 1, n),
    }
