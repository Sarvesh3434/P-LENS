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
    output_dir: str = "results"  # relative to project_root, overridable by runners
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
    rewire_targets_per_edge: int = 8  # 0 = all targets; >0 keeps REWIRE enumeration O(m k)
    connectivity_required: bool = True  # [DIAGRAM]+[ASSUMPTION] stay connected
    resilience_threshold: float = 0.15
    lambda_target: float = 2.0  # local connectivity target [ASSUMPTION]

    # --- simulation [SIM] ---
    simulation_steps: int = 120
    traffic_pattern: str = "stationary"  # stationary | shift | periodic | bursty
    failure_mode: str = "random"  # random | targeted | multi | recovery
    sla_latency_ms: float = 2000.0  # SLA threshold for tail/SLA metrics [SIM] (queue term is normalised to <=1 s)
    sla_loss: float = 0.05
    traffic_intensity: float = 0.30  # target median utilization after scaling
    failure_rate_edge: float = 0.01
    failure_rate_node: float = 0.001
    repair_steps: int = 5
    seed: int = 7
    n_seeds: int = 3  # demo default; the study runner overrides to 30
    mm1_service_rate: float = 2.0e6  # per-link service rate = nominal bandwidth [SIM]
    routing_alpha: float = 2.0  # congestion sensitivity of routing weights [ASSUMPTION]
    hop_epsilon: float = 1e-4  # tiny per-hop cost, breaks weight ties [ASSUMPTION]

    # --- P-LENS [ASSUMPTION] ---
    delay_k: int = 5
    mutation_budget_initial: float = 3.0
    mutation_budget_min: float = 0.0
    mutation_budget_max: float = 8.0
    budget_mode: str = "adaptive"  # adaptive | fixed | legacy
    budget_refill: float = 4.5
    budget_base_refill: float = 0.5  # per-step stipend; debt must justify spending [ASSUMPTION]
    budget_uncertainty_z: float = 1.0
    budget_overload_window: int = 8
    budget_overload_threshold: float = 0.65
    budget_overload_damping: float = 0.5
    budget_failure_damping: float = 0.35
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
    # --- debt: dependency-aware structural maladaptation over time [ASSUMPTION] ---
    debt_tolerance: float = 0.35  # deadband: only excess pressure accumulates
    debt_accumulation: float = 0.18  # eta: increment toward saturation
    debt_decay: float = 0.12  # gamma: leak when the pressure is resolved
    debt_persist_beta: float = 0.75  # persistence evidence EMA
    debt_duration_tau: float = 3.0  # duration saturation constant
    debt_relief: float = 0.20  # max relief per successful mutation
    debt_failure_penalty: float = 0.04  # debt added by an unsuccessful mutation
    debt_gain_relief: float = 0.50  # relief scales with the counterfactual gain
    criticality_floor: float = 0.50  # criticality weight floor
    pressure_ema: float = 0.4
    credit_lambda: float = 0.7
    cooldown_steps: int = 4
    max_mutations_per_step: int = 4

    # --- debt propagation, rate and modification suppression [NEW] ---
    debt_propagation: float = 0.35  # kappa: weight of neighbour inadequacy
    debt_propagation_tolerance: float = 0.05  # rho: deadband on propagated debt
    debt_rate_gain: float = 0.10  # eta_r^+: sensitivity to worsening inadequacy
    debt_rate_decay: float = 0.10  # eta_r^-: sensitivity to improving inadequacy
    debt_dependency_coupling: float = 0.50  # flow-weighted dependency strength
    debt_relief_propagation: float = 0.25  # diffusion of relief to dependants
    debt_mod_suppression: float = 1.50  # mu: damp debt while topology is changing
    debt_mod_window: int = 6  # steps a mutation keeps its neighbourhood "in flight"

    # --- mutation economics: primal-dual price + resource account [NEW] ---
    budget_price_initial: float = 0.35  # lambda_0: shadow price of one cost unit
    budget_price_min: float = 0.0
    budget_price_max: float = 3.0
    budget_dual_step: float = 0.08  # alpha_0 in lambda <- proj(lambda + alpha*residual)
    budget_price_scale: float = 1.0  # residual -> price conversion
    budget_target_utilization: float = 0.75  # target spend fraction of the account
    budget_marginal_gain: float = 0.60  # kappa_mu: refill from marginal utility
    budget_marginal_ref: float = 0.05  # MU_0: utility above which refill grows
    budget_churn_penalty: float = 0.20  # price increase per unit churn
    budget_price_refill_damping: float = 0.15  # high price slows account growth
    budget_dual_decay: float = 0.995  # forgetting of stale price pressure

    # --- interaction-aware selection: higher-order dependencies [NEW] ---
    interaction_weight: float = 0.35
    interaction_radius: int = 1  # neighbourhood for complementarity
    interaction_order: int = 3  # 1 = marginal, 2 = pairwise, 3 = triple/multi-way
    interaction_triple_weight: float = 0.15  # w_3 on third-order factors
    joint_window: int = 64  # lazy-greedy look-ahead window
    joint_pair_window: int = 8  # pair look-ahead width for joint moves
    joint_pair_gain_margin: float = 0.05  # pair must beat best singleton by this
    joint_local_search: bool = True  # bounded joint/look-ahead improvement
    joint_conflict_propagation: bool = True  # infeasible conflict closure

    # --- counterfactual evaluation / attribution [NEW] ---
    eval_mode: str = "replay_counterfactual"  # replay_counterfactual | shadow | before_after | immediate
    cf_bootstrap: int = 0  # matched-demand bootstrap replicates (0 = cheap stepwise paired uncertainty)
    cf_z: float = 1.0  # lower-confidence-bound multiplier (kept for compatibility)
    cf_min_effect: float = 0.005  # below this the credit update is suppressed
    cf_min_confidence: float = 0.20  # uncertainty gate for credit updates
    cf_perturb_sigma: float = 0.15  # lognormal OD perturbation for bootstrap
    attribution_mode: str = "direct_indirect"  # total | direct_indirect
    cf_confidence_alpha: float = 0.05  # 1-alpha Student-t / Hoeffding confidence level
    cf_lcb_method: str = "t"  # t | hoeffding | z
    cf_drift_threshold: float = 0.35  # total-variation drift above which replay is unreliable
    cf_drift_weight: float = 0.50  # weight of metric drift in the drift score
    cf_credit_drift_penalty: float = 1.0  # credit multiplier = 1 - penalty*drift

    # --- unified optimizer [NEW] ---
    use_unified_optimizer: bool = True  # compose debt/economics/interaction/attribution
    unified_debt_weight: float = 0.35  # beta_D: debt-relief potential in the objective
    unified_credit_weight: float = 0.60  # beta_C: counterfactual value in the objective
    unified_uncertainty_weight: float = 0.25  # rho: risk of an uncertain mutation
    unified_churn_weight: float = 0.20  # mu: expected churn penalty
    unified_risk_weight: float = 0.20  # nu: structural risk penalty

    policy: str = "plens"  # plens | static | random | greedy | threshold | fixed_budget | adaptive_budget | optimization | bandit
    ablation: str = "none"  # none | no_ml | no_debt | no_credit | no_budget | no_uncertainty | no_interaction | fixed_score | no_counterfactual | no_evaluation
    threshold_trigger: float = 0.55  # threshold baseline trigger on max PP
    bandit_ucb_c: float = 0.6  # exploration constant for the learning baseline

    # --- ML [ML] ---
    window_length_L: int = 12
    ml_horizon: int = 1  # forecast lead in windows (one 400-flow window ahead)
    flows_per_window: int = 400
    gru_hidden: int = 64
    gru_layers: int = 2
    dropout: float = 0.2
    epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 5
    use_class_weights: bool = True
    ml_blend_beta: float = 0.3
    ml_prior_attack: float = 0.20
    ml_tolerance_gain: float = 0.35  # lower debt tolerance when attack is likely
    ml_budget_gain: float = 0.25  # refill urgency from p(attack)
    ml_uncertainty_damping: float = 0.25  # refill damping from forecast entropy
    stress_scales: Dict[str, float] = field(
        default_factory=lambda: {"Normal": 0.85, "Attack": 1.25}
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
        d = self.project_root / self.output_dir
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






