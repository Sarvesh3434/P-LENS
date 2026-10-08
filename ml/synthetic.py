"""[ML][SIM] Synthetic traffic-window generator for dataset="synthetic".

Why: the pipeline should be runnable end-to-end without the external CIC-IDS2017
/ UNSW-NB15 files (e.g. on a fresh machine, in CI, or for debugging). The
generator produces per-window aggregate features with the same columns and the
same column order as FEATURE_NAMES, plus a per-window demand matrix, so every
downstream consumer (scaler, stress map, GRU, simulation) works unchanged.
[ASSUMPTION] This is a stand-in traffic model, not a claim about real datasets:

- background flows arrive as a Poisson process whose rate follows a diurnal
  sinusoid (peak mid-window) when cfg.synthetic_diurnal is on;
- each flow's packets/bytes are lognormal (heavy-tailed, like real flow-length
  distributions);
- a fraction of windows get an attack burst (port-scan-like: many tiny flows to
  many distinct destination ports), which raises flow_count/pkt_rate and thus
  the composite stress label;
- endpoints are drawn uniformly (uniform CRC-free — no hashing needed here).

Labels are NOT produced here: stress classes still come from the training-ECDF
tertiles in ml.preprocessing.fit_stress_map, exactly as for real data. [ML]
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from experiments.config import Config
from ml.data_loader import FEATURE_NAMES


def synthetic_windows(
    cfg: Config,
    n_windows: int | None = None,
    seed: int = 1234,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Generate (feats, demands, metas) shaped exactly like the real loaders.

    feats:  (W, len(FEATURE_NAMES)) float64 — same columns as FEATURE_NAMES
    demands: (W, n, n) float64 — per-window demand matrices
    metas:  list of {"attack_frac", "n_flows"} dicts, same keys as real loaders
    """
    rng = np.random.default_rng(seed)
    W = n_windows if n_windows is not None else cfg.synthetic_windows
    n = cfg.network_size
    feats = np.zeros((W, len(FEATURE_NAMES)), dtype=np.float64)
    demands = np.zeros((W, n, n), dtype=np.float64)
    metas: List[dict] = []

    t = np.arange(W, dtype=np.float64)
    if cfg.synthetic_diurnal:
        # one "day" every 96 windows; peak at 1/4 into the day [ASSUMPTION]
        diurnal = 0.6 + 0.4 * np.sin(2.0 * np.pi * (t % 96) / 96.0)
    else:
        diurnal = np.ones(W)

    attack_windows = rng.random(W) < cfg.synthetic_attack_frac

    for w in range(W):
        lam = 300.0 * diurnal[w]  # background flows per window [ASSUMPTION]
        n_flows = int(rng.poisson(lam))
        if attack_windows[w]:
            n_flows += int(rng.integers(200, 400))  # scan burst [ASSUMPTION]

        # flow sizes: lognormal, heavy-tailed [ASSUMPTION]
        fwd_pkts = rng.lognormal(mean=2.0, sigma=1.0, size=n_flows)
        bwd_pkts = fwd_pkts * rng.lognormal(mean=-0.3, sigma=0.5, size=n_flows)
        fwd_bytes = fwd_pkts * rng.lognormal(mean=6.5, sigma=0.4, size=n_flows)
        bwd_bytes = bwd_pkts * rng.lognormal(mean=6.0, sigma=0.4, size=n_flows)
        dur = rng.lognormal(mean=5.0, sigma=1.0, size=n_flows)

        # endpoints: uniform over the simulated network's nodes [SIM]
        src = rng.integers(0, n, size=n_flows)
        dst = (src + rng.integers(1, n, size=n_flows)) % n

        if attack_windows[w]:
            # port-scan-like: many distinct dst ports, tiny flows [ASSUMPTION]
            n_scan = min(n_flows, 300)
            fwd_pkts[:n_scan] = 1.0
            bwd_pkts[:n_scan] = 0.0
            fwd_bytes[:n_scan] = 60.0
            bwd_bytes[:n_scan] = 0.0
            dur[:n_scan] = 0.05

        pkts = fwd_pkts + bwd_pkts
        byts = fwd_bytes + bwd_bytes
        ports = rng.integers(1, 65536, size=n_flows)
        unique_ports_frac = len(np.unique(ports)) / max(n_flows, 1)

        feats[w] = [
            float(n_flows),
            float(pkts.sum()),
            float(byts.sum()),
            float(dur.mean()),
            float(fwd_pkts.mean()),
            float(bwd_pkts.mean()),
            float(fwd_bytes.mean()),
            float(bwd_bytes.mean()),
            float(unique_ports_frac),
            float(attack_windows[w]),  # attack_flag: 1.0 iff an attack burst fired [ML]
        ]

        # demand matrix: bytes between endpoint pairs, symmetrised [SIM]
        d = np.zeros((n, n), dtype=np.float64)
        np.add.at(d, (src, dst), byts)
        d = d + d.T
        np.fill_diagonal(d, 0.0)
        demands[w] = d
        metas.append(
            {
                "attack_frac": float(attack_windows[w]),
                "n_flows": int(n_flows),
            }
        )
    return feats, demands, metas
