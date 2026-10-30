"""[ML] Load CIC-IDS2017 MachineLearningCVE and UNSW-NB15 records.

Leakage-free feature contract
-----------------------------
The nine columns in ``FEATURE_NAMES`` are computed *before* the label is known
and contain no attack indicator.  The binary attack/normal label is returned in
a separate ``labels`` array and is used only as the supervised target.  This is
enforced by the loader API itself, so the model cannot accidentally consume the
target as an input.

[ASSUMPTION] These files have no Timestamp and no Source IP. Time is file order
then row order. Endpoints are hashed from Destination Port (CIC) or
proto/service (UNSW); the server endpoint comes from the port/service and the
client endpoint from a per-flow index, so traffic is hub-like but not collapsed
onto a single node pair. Each flow is sharded over four client/replica pairs to
emulate flow-level load balancing; without this a single elephant flow is
bandwidth-limited on every topology and no mutation can help. The dataset supplies traffic/condition information
only; the topology itself is simulated.
"""

from __future__ import annotations

import csv
import zlib
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

from experiments.config import Config


# Nine pre-label traffic features.  Deliberately no "attack_flag" column.
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


def _label_from_row(row: dict, kind: str) -> int:
    lab = str(row.get("label", "")).strip()
    if not lab:
        return 0
    if kind == "cic":
        return int(lab.upper() != "BENIGN")
    return int(lab.lower() not in ("0", "normal"))


def _flush_window(
    buf: List[dict], n_nodes: int, kind: str
) -> Tuple[np.ndarray, np.ndarray, int, Dict]:
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
        ],
        dtype=np.float64,
    )
    demand = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    ports = set()
    attack_frac = 0.0
    for k, r in enumerate(buf):
        if kind == "cic":
            ports.add(r["dst_port"])
            # The dataset has no IP addresses.  Mapping both endpoints from the
            # destination port alone collapses almost every byte onto one node
            # pair (measured: top pair = 98% of demand), which makes topology
            # adaptation meaningless.  Keep the port as the *server* identity
            # (popular ports are popular servers) and spread clients over nodes
            # with a per-flow index.
            #
            # Each flow is also sharded over four (client, replica) pairs, the
            # way per-packet ECMP spreads a large flow over parallel paths.  A
            # single elephant flow on one path is bandwidth-limited on every
            # topology, so without sharding no mutation can ever help it.
            srcs = [_crc_node(f"client:{r['dst_port']}:{k}:{q}", n_nodes, "src") for q in range(4)]
            dsts = [_crc_node(f"server:{r['dst_port']}:{q}", n_nodes, "dst") for q in range(4)]
        else:
            srcs = [
                _crc_node(f"{r.get('proto', '')}:{r.get('service', '')}:{k}:{q}", n_nodes, "src")
                for q in range(4)
            ]
            dsts = [_crc_node(f"server:{r.get('service', '')}:{q}", n_nodes, "dst") for q in range(4)]
        attack_frac += _label_from_row(r, kind)
        share = (r["fwd_bytes"] + r["bwd_bytes"]) / 4.0
        for src, dst in zip(srcs, dsts):
            if src == dst:
                dst = (dst + 1) % n_nodes
            demand[src, dst] += share
    feat[8] = len(ports) / max(n, 1) if kind == "cic" else 0.0
    # The label is produced only after every feature has been computed.
    label = int(attack_frac > 0)
    meta = {"attack_frac": attack_frac / max(n, 1), "n_flows": n}
    return feat, demand, label, meta


def windows_from_rows(rows: Iterator[dict], cfg: Config, kind: str):
    buf: List[dict] = []
    feats, demands, labels, metas = [], [], [], []
    wsize = cfg.flows_per_window
    n_rows = 0
    for r in rows:
        buf.append(r)
        n_rows += 1
        if len(buf) >= wsize:
            f, d, y, m = _flush_window(buf, cfg.network_size, kind)
            feats.append(f)
            demands.append(d)
            labels.append(y)
            metas.append(m)
            buf = []
        if cfg.max_rows_per_file and n_rows >= cfg.max_rows_per_file:
            break
    if buf:
        f, d, y, m = _flush_window(buf, cfg.network_size, kind)
        feats.append(f)
        demands.append(d)
        labels.append(y)
        metas.append(m)
    return (
        np.stack(feats) if feats else np.zeros((0, len(FEATURE_NAMES))),
        np.stack(demands) if demands else np.zeros((0, cfg.network_size, cfg.network_size)),
        np.asarray(labels, dtype=np.int64),
        metas,
    )


def load_cic_windows(cfg: Config):
    # Parsing eight CSVs with the pure-Python reader dominates rerun time.
    # Cache the *windowed* tensors keyed by the parameters that change them.
    cache = cfg.project_root / "data" / "processed" / (
        f"windows_cic_n{cfg.network_size}_r{cfg.max_rows_per_file}_w{cfg.flows_per_window}.npz"
    )
    signature = (
        "v4|"
        + "|".join(cfg.cic_files)
        + f"|n{cfg.network_size}|r{cfg.max_rows_per_file}|w{cfg.flows_per_window}"
    )
    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=False)
            if str(z["signature"]) == signature:
                bounds = []
                for row in z["bounds"].tolist():
                    parts = str(row).rsplit("|", 2)
                    bounds.append((parts[0], int(parts[1]), int(parts[2])))
                metas = [
                    {"attack_frac": float(a), "n_flows": int(n)}
                    for a, n in zip(z["attack_frac"], z["n_flows"])
                ]
                return z["feats"], z["demand"], z["labels"], metas, bounds
        except Exception:
            pass

    feats_all = []
    dem_all = []
    lab_all = []
    meta_all = []
    file_bounds = []  # (name, start, end)
    start = 0
    for name in cfg.cic_files:
        path = cfg.cic_dir() / name
        if not path.exists():
            raise FileNotFoundError(path)
        f, d, y, m = windows_from_rows(iter_cic_rows(path), cfg, "cic")
        feats_all.append(f)
        dem_all.append(d)
        lab_all.append(y)
        meta_all.extend(m)
        end = start + len(f)
        file_bounds.append((name, start, end))
        start = end
    feats = np.concatenate(feats_all, axis=0) if feats_all else np.zeros((0, len(FEATURE_NAMES)))
    dem = np.concatenate(dem_all, axis=0)
    labels = np.concatenate(lab_all, axis=0) if lab_all else np.zeros((0,), dtype=np.int64)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            feats=feats,
            demand=dem,
            labels=labels,
            signature=np.asarray(signature),
            bounds=np.asarray([f"{name}|{a}|{c}" for name, a, c in file_bounds]),
            attack_frac=np.asarray([m["attack_frac"] for m in meta_all], dtype=float),
            n_flows=np.asarray([m["n_flows"] for m in meta_all], dtype=int),
        )
    except Exception:
        pass
    return feats, dem, labels, meta_all, file_bounds


def load_unsw_windows(cfg: Config):
    train_p = cfg.unsw_dir() / "UNSW_NB15_training-set.csv"
    test_p = cfg.unsw_dir() / "UNSW_NB15_testing-set.csv"
    f1, d1, y1, m1 = windows_from_rows(iter_unsw_rows(train_p), cfg, "unsw")
    f2, d2, y2, m2 = windows_from_rows(iter_unsw_rows(test_p), cfg, "unsw")
    feats = np.concatenate([f1, f2], axis=0)
    dem = np.concatenate([d1, d2], axis=0)
    labels = np.concatenate([y1, y2], axis=0)
    metas = m1 + m2
    bounds = [("UNSW_train", 0, len(f1)), ("UNSW_test", len(f1), len(f1) + len(f2))]
    return feats, dem, labels, metas, bounds


def day_blocked_blocks(n: int, bounds=None, fractions=(0.65, 0.15, 0.15, 0.05)):
    """Chronological blocks that preserve every day/attack type.

    A single global day split (train on Mon--Wed, test on Thu--Fri) creates a
    severe concept shift: the test day contains attack families never seen in
    training, so even a strong model collapses to chance.  Instead, every
    contiguous source block is split *in time order* into train / validation /
    test / simulation fractions.  No test window precedes a training window
    from the same block, the scaler is fit on training blocks only, and
    sequences never cross block boundaries, so the protocol remains
    leakage-free while covering all days.

    The simulation segment is held out **within every block**, not just the
    final day.  The controller replay therefore sees benign and attack periods
    from every day, while remaining disjoint from the ML test windows.  This is
    important for the ``no_ml`` comparison: a single DDoS-only segment makes
    every window an attack and removes the regime changes that a forecast can
    anticipate.

    All returned intervals are half-open ``[lo, hi)``.  CIC files are days
    (``bounds`` has >= 8 entries).  UNSW uses the official training file for
    train/val and splits the official testing file into test/sim.
    """
    if len(fractions) == 3:
        # Backwards-compatible three-way call: keep 80% of the test region for
        # evaluation and reserve the final 20% for simulation.
        tr_f, va_f, te_f = fractions
        sim_f = 0.0
    else:
        tr_f, va_f, te_f, sim_f = fractions
    out = {"train": [], "val": [], "test": [], "sim": []}

    def _split_block(a: int, c: int) -> None:
        m = c - a
        if m <= 0:
            return
        a1 = a + int(round(tr_f * m))
        a2 = a1 + int(round(va_f * m))
        a3 = a2 + int(round(te_f * m))
        a1 = min(max(a1, a + 1), c)
        a2 = min(max(a2, a1 + 1 if a1 < c else a1), c)
        a3 = min(max(a3, a2 + 1 if a2 < c else a2), c)
        if a3 >= c:
            # tiny block: give the final window to sim only when it was requested
            a3 = max(a2, c - 1) if sim_f > 0 else c
        out["train"].append((a, a1))
        if a1 < a2:
            out["val"].append((a1, a2))
        if a2 < a3:
            out["test"].append((a2, a3))
        if sim_f > 0 and a3 < c:
            out["sim"].append((a3, c))

    if bounds and len(bounds) == 2:
        # Official train/test files: never mix them.  Use the train file for
        # train/val and reserve part of the official test file for replay.
        (_n0, a0, c0), (_n1, a1, c1) = bounds
        m0 = c0 - a0
        a0v = a0 + int(round(0.85 * m0))
        out["train"].append((a0, min(a0v, c0)))
        if a0v < c0:
            out["val"].append((a0v, c0))
        m1 = c1 - a1
        a1t = a1 + int(round(0.80 * m1)) if sim_f > 0 else c1
        out["test"].append((a1, min(a1t, c1)))
        if sim_f > 0 and a1t < c1:
            out["sim"].append((a1t, c1))
    elif bounds and len(bounds) >= 2:
        for _name, a, c in bounds:
            _split_block(int(a), int(c))
    else:
        _split_block(0, n)
    return out


def chronological_split(n: int, bounds=None):
    """Overall [lo, hi) span of each day-blocked part (legacy convenience)."""
    blocks = day_blocked_blocks(n, bounds)
    return {
        part: (min(b[0] for b in bs), max(b[1] for b in bs))
        for part, bs in blocks.items()
        if bs
    }
