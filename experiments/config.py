"""[ASSUMPTION] Central configuration. Every parameter has a documented reason."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# Datasets live in the sibling CN_Project_ML_Training workspace (user-provided).
# Files sit directly in DataSets/ (flat layout); nested layouts are handled in
# cic_dir()/unsw_dir() below. [ASSUMPTION]
DEFAULT_DATASET_ROOT = Path(
    r"C:\Users\sarve\Documents\Project\CN_Project_ML_Training\DataSets"
)


@dataclass
class Config:
    # --- paths ---
    dataset_root: Path = DEFAULT_DATASET_ROOT
    project_root: Path = Path(__file__).resolve().parents[1]
    # cic | unsw | synthetic
    dataset: str = "cic"
    # [ASSUMPTION] CIC MachineLearningCVE has no Timestamp/IP; weekday file order is used as time.

    # --- network [SIM] ---
    network_size: int = 24  # small enough for a live demo, large enough for mutations
    initial_topology: str = "small_world"  # ring | erdos_renyi | barabasi_albert | small_world
    er_p: float = 0.18
    ba_m: int = 2
    sw_k: int = 4
    sw_p: float = 0.15
    node_capacity: float = 1.0e7  # bytes/step [SIM]
    default_bandwidth: float = 2.0e6  # bytes/step per link [SIM]
    default_prop_latency: float = 0.002  # seconds [SIM]
    max_degree: int = 8
    min_degree: int = 2
    connectivity_required: bool = True  # [DIAGRAM]+[ASSUMPTION] stay connected
    resilience_threshold: float = 0.15
    lambda_target: float = 2.0  # local connectivity target [ASSUMPTION]

    # --- simulation [SIM] ---
    simulation_steps: int = 120
    traffic_intensity: float = 0.55  # target median utilization after scaling
    failure_rate_edge: float = 0.01
    failure_rate_node: float = 0.001
    repair_steps: int = 5
    seed: int = 7
    n_seeds: int = 3  # demo default; paper-style runner uses 30
    mm1_service_rate: float = 1.2e6  # packets-or-bytes proxy / step [SIM]
    routing_alpha: float = 2.0  # congestion sensitivity of routing weights [ASSUMPTION]
    hop_epsilon: float = 1e-4  # tiny per-hop cost, breaks weight ties [ASSUMPTION]

    # --- P-LENS [ASSUMPTION] ---
    delay_k: int = 5
    mutation_budget_initial: float = 3.0
    mutation_budget_min: float = 0.0
    mutation_budget_max: float = 8.0
    budget_refill: float = 1.0
    eta_g: float = 0.5
    eta_s: float = 0.4
    mutation_cost: Dict[str, float] = field(
        default_factory=lambda: {"ADD": 1.0, "PRUNE": 0.8, "REWIRE": 1.2}
    )
    pressure_weights: Dict[str, float] = field(
        default_factory=lambda: {"D": 0.25, "F": 0.25, "C": 0.25, "R": 0.25}
    )
    congestion_weights: Dict[str, float] = field(
        default_factory=lambda: {"u": 0.4, "l": 0.3, "q": 0.2, "p": 0.1}
    )
    gain_weights: Dict[str, float] = field(
        default_factory=lambda: {"u": 0.4, "r": 0.4, "s": 0.2}
    )
    field_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "need": 1.0,
            "debt": 0.8,
            "benefit": 0.7,
            "credit": 0.6,
            "cost": 0.5,
            "penalty": 1.2,
        }
    )
    debt_decay: float = 0.9
    debt_accumulation: float = 0.2
    debt_relief: float = 0.15
    pressure_ema: float = 0.4
    credit_lambda: float = 0.7
    cooldown_steps: int = 4
    max_mutations_per_step: int = 4
    eval_mode: str = "counterfactual"  # counterfactual | before_after
    policy: str = "plens"  # plens | static | random | greedy
    ablation: str = "none"  # none | no_ml | no_debt | no_credit | no_budget

    # --- ML [ML] ---
    window_length_L: int = 12
    flows_per_window: int = 400
    gru_hidden: int = 64
    gru_layers: int = 2
    dropout: float = 0.2
    epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 5
    ml_blend_beta: float = 0.3  # mix forecast into F_i, C_i only
    stress_scales: Dict[str, float] = field(
        default_factory=lambda: {"Low": 0.7, "Medium": 1.0, "High": 1.3}
    )
    max_train_windows: int = 4000  # cap so CPU training stays in demo territory
    max_rows_per_file: int = 200000  # [ASSUMPTION] stream cap per CIC/UNSW file (200k => ~500 windows/file)

    # synthetic traffic generator (dataset="synthetic") [SIM]
    synthetic_windows: int = 1200  # number of windows to generate
    synthetic_nodes_hint: int = 8  # hotspot nodes in the demand matrix
    synthetic_diurnal: bool = True  # superimpose a daily sinusoid on the load
    synthetic_attack_frac: float = 0.15  # fraction of windows with attack bursts

    # CIC weekday files in chronological order [ASSUMPTION: file order = time]
    cic_files: List[str] = field(
        default_factory=lambda: [
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        ]
    )

    def artifacts_dir(self) -> Path:
        d = self.project_root / "results"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def model_path(self) -> Path:
        return self.artifacts_dir() / "model.pt"

    def scaler_path(self) -> Path:
        return self.artifacts_dir() / "scaler.pkl"

    def processed_path(self) -> Path:
        p = self.project_root / "data" / "processed"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"windows_{self.dataset}.npz"

    def cic_dir(self) -> Path:
        """CIC-IDS2017 MachineLearningCVE folder; supports flat and nested layouts."""
        nested = self.dataset_root / "MachineLearningCSV" / "MachineLearningCVE"
        return nested if nested.exists() else self.dataset_root

    def unsw_dir(self) -> Path:
        """UNSW-NB15 folder; supports flat and nested layouts."""
        if (self.dataset_root / "UNSW_NB15_training-set.csv").exists():
            return self.dataset_root
        return self.dataset_root / "OneDrive_1_19-9-2026"


def default_config(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
