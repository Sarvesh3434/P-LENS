"""[DIAGRAM] P-LENS controller: PP → PD → B → MF → mutation → MC, plus feedback F1–F5.

Loop order per step (matches the architecture image left→right, then the feedback row):
 1. [SIM] apply failures/recovery          → Dynamic Network Environment
 2. [ML]  optional stress forecast          → blended demand for F_i, C_i only
 3. [SIM] route & load, snapshot metrics    → observation
 4. [DIAGRAM] monitor D, F, C, R            → State & Structural Drift Monitor
 5. [DIAGRAM] PP_i                          → Plasticity Pressure Estimator
 6. [DIAGRAM] PD_i (after selection, with relief)
 7. [DIAGRAM] B_t                           → Mutation Budget Controller (gate)
 8. [DIAGRAM] MF_ij^op                      → Topology Mutation Field Generator
 9. [DIAGRAM] constraint-aware selection    → ADD / PRUNE / REWIRE → G_{t+1}
10. [DIAGRAM] delayed evaluation at t+k     → Utility/Resilience/Stability
11. [DIAGRAM] MC_ij update                  → Mutation Credit (feeds next MF [ASSUMPTION])
12. Feedback row: PD(t+1), B(t+1)           → F3, F4; PP carries via EMA (F5)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from experiments.config import Config
from network.graph import EvolvingNetwork, canon
from network.failures import apply_failures
from network.traffic import route_and_load, scale_demand
from plens.indicators import compute_indicators
from plens.pressure import plasticity_pressure
from plens.debt import update_debt, relief_for_mutation
from plens.budget import update_budget
from plens.mutation_field import generate_field
from plens.selector import select_mutations
from plens.evaluation import evaluate_pending
from plens.credit import update_credit


class PLENSController:
    def __init__(self, cfg: Config, net: EvolvingNetwork, ml_predictor=None, base_rng=None):
        self.cfg = cfg
        self.net = net
        self.ml = ml_predictor
        self.pp = np.zeros(cfg.network_size)
        self.pd = np.zeros(cfg.network_size)
        self.budget = cfg.mutation_budget_initial
        self.mc: Dict[tuple, float] = {}
        self.cooldown: Dict[tuple, int] = {}
        self.pending: List[dict] = []
        self.history: List[dict] = []
        self.rejection_log: List[dict] = []
        # [DIAGRAM] G_t → G_(t+1) evolution trace for visualization [ASSUMPTION: implied by loop]
        self.topology_snapshots: List[list] = [[list(e) for e in net.G.edges()]]
        self.rng = base_rng if base_rng is not None else np.random.default_rng(cfg.seed)
        self._ablate()

    def _ablate(self) -> None:
        a = self.cfg.ablation
        self.use_ml = self.cfg.policy == "plens" and a != "no_ml" and self.ml is not None
        self.use_debt = a != "no_debt"
        self.use_credit = a != "no_credit"
        self.use_budget = a != "no_budget"
        if not self.use_budget:
            self.budget = self.cfg.mutation_budget_max

    def step(
        self,
        t: int,
        demand: np.ndarray,
        feature_window_hist: Optional[np.ndarray] = None,
    ) -> dict:
        cfg = self.cfg
        net = self.net
        net.step = t
        apply_failures(net)

        # [ML] forecast stress for the next window; it modulates the *offered-load
        # intensity* target for this step (not the demand shape). [ASSUMPTION]
        # (An earlier version scaled the demand matrix itself, but scale_demand()
        # renormalizes total demand to traffic_intensity x usable_bw, so a scalar
        # matrix scale cancelled out exactly and the forecast had no effect.)
        ml_pred = None
        target_intensity = cfg.traffic_intensity
        if self.use_ml and feature_window_hist is not None and t >= cfg.window_length_L:
            ml_pred = self.ml.predict_stress(feature_window_hist)
            stress_scale = cfg.stress_scales.get(ml_pred["label"], 1.0)
            # blend the intensity target: beta=0 → measured target, beta=1 → forecast
            target_intensity = (
                (1.0 - cfg.ml_blend_beta) * cfg.traffic_intensity
                + cfg.ml_blend_beta * cfg.traffic_intensity * stress_scale
            )

        net.set_demand(scale_demand(demand, net, target_intensity))
        metrics = route_and_load(net)

        # [DIAGRAM] Monitor → D, F, C, R (per node) [ASSUMPTION formulas]
        ind = compute_indicators(net, net.demand)
        # [DIAGRAM] PP_i [ASSUMPTION weighted mix, EMA smooth]
        self.pp = plasticity_pressure(ind, cfg, self.pp)

        for k in list(self.cooldown):
            self.cooldown[k] -= 1
            if self.cooldown[k] <= 0:
                del self.cooldown[k]

        mutations: List[dict] = []
        spent = 0.0
        rejected: List[dict] = []
        if cfg.policy == "static":
            mutations = []
        elif cfg.policy == "random":
            mutations = self._random_policy()
        elif cfg.policy == "greedy":
            mutations = self._greedy_policy()
        else:
            field = generate_field(
                net, self.pp, self.pd if self.use_debt else np.zeros_like(self.pd),
                self.mc, self.budget, cfg, use_credit=self.use_credit,
            )
            mutations, rejected = select_mutations(net, field, self.budget, cfg, self.cooldown)
            spent = sum(m["budget_cost"] for m in mutations)

        mutation_records: List[dict] = []
        for m in mutations:
            d = m.get("detail") or {}
            # freeze 'before' metrics for the before_after evaluation mode
            d["before"] = dict(net.last_metrics)
            if d.get("added"):
                self.cooldown[canon(*d["added"])] = cfg.cooldown_steps
            if d.get("removed"):
                self.cooldown[canon(*d["removed"])] = cfg.cooldown_steps
            rec_m = {
                "timestep": t,
                "mutation": m["op"],
                "source_node": m["i"],
                "target_node": m.get("l", m["j"]),
                "removed_edge": d.get("removed"),
                "added_edge": d.get("added"),
                "score": m.get("score"),
                "budget_cost": m.get("budget_cost"),
                "reason": m.get("reason"),
                "delayed_eval": None,
            }
            mutation_records.append(rec_m)
            # 'record' is the live history entry; delayed_eval is written into it
            # at t+k so the saved history reflects the delayed evaluation [F2]
            self.pending.append({"t_eval": t + cfg.delay_k, "mutation": m, "record": rec_m})

        # --- [DIAGRAM] delayed evaluation F2 ---
        gains = []
        stab_costs = []
        relief = np.zeros(cfg.network_size)
        still = []
        for item in self.pending:
            if item["t_eval"] > t:
                still.append(item)
                continue
            m = item["mutation"]
            detail = m.get("detail") or {}
            ev = {"G": 0.0, "utility": 0.0, "resilience": 0.0, "stability_cost": 0.0}
            if detail:
                ev = evaluate_pending(net, detail, cfg, n_mutations=len(mutations))
            gains.append(max(ev["G"], 0.0))
            stab_costs.append(ev["stability_cost"])
            endpoints = []
            if detail.get("added"):
                endpoints += detail["added"]
                update_credit(self.mc, *detail["added"], ev["G"], cfg.credit_lambda)
            if detail.get("removed"):
                endpoints += detail["removed"]
                update_credit(self.mc, *detail["removed"], ev["G"], cfg.credit_lambda)
            # neighbours also get relief [ASSUMPTION per prompt default 9]
            touched = set(endpoints)
            for node in list(touched):
                touched |= net.neighbours(node)
            success = ev["G"] > 0
            relief += relief_for_mutation(cfg.network_size, list(touched), cfg, success)
            m["delayed_eval"] = ev
            item["record"]["delayed_eval"] = ev  # keep the history record in sync [F2]
        self.pending = still
        relief = np.clip(relief, 0.0, cfg.debt_relief)

        # --- feedback row: PD(t+1) [F3] and B(t+1) [F4] ---
        self.pd = update_debt(self.pd, self.pp, relief, cfg, enabled=self.use_debt)
        self.budget = update_budget(
            self.budget,
            spent,
            float(self.pd.mean()),
            float(np.mean(gains) if gains else 0.0),
            float(np.mean(stab_costs) if stab_costs else 0.0),
            cfg,
            enabled=self.use_budget,
        )

        # F1 structural loop: G_{t+1} is now the environment the next step observes
        self.topology_snapshots.append([list(e) for e in net.G.edges()])

        rec = {
            "t": t,
            "metrics": metrics,
            "pp_mean": float(self.pp.mean()),
            "pd_mean": float(self.pd.mean()),
            "budget": float(self.budget),
            "budget_spent": float(spent),
            "mutations": mutation_records,
            "rejections": [
                {"op": r.get("op"), "i": r.get("i"), "j": r.get("j"), "reason": r.get("reason")}
                for r in rejected
            ],
            "ml": ml_pred,
            "n_add": sum(1 for m in mutations if m["op"] == "ADD"),
            "n_prune": sum(1 for m in mutations if m["op"] == "PRUNE"),
            "n_rewire": sum(1 for m in mutations if m["op"] == "REWIRE"),
            "mc_mean": float(np.mean(list(self.mc.values())) if self.mc else 0.0),
            "indicators_mean": {k: float(v.mean()) for k, v in ind.items()},
        }
        self.history.append(rec)
        if rejected:
            self.rejection_log.append({"t": t, "rejected": rec["rejections"]})
        return rec

    def _random_policy(self) -> List[dict]:
        """Baseline: random-adaptive under the SAME hard constraints and max count. [ASSUMPTION]

        Uses a dedicated baseline RNG stream (seeded from cfg.seed) so that
        consuming random candidates does not shift the failure/traffic streams
        shared across policies — common random numbers stay comparable. [SIM]
        """
        from plens.mutation_field import generate_field
        from plens import mutations as muts

        cfg = self.cfg
        field = generate_field(
            self.net, self.pp, self.pd, self.mc, self.budget, cfg, use_credit=False
        )
        if not hasattr(self, "_base_rng"):
            self._base_rng = np.random.default_rng(cfg.seed)
        self._base_rng.shuffle(field)
        chosen = []
        spent = 0.0
        for cand in field:
            if len(chosen) >= cfg.max_mutations_per_step:
                break
            cost = cfg.mutation_cost[cand["op"]]
            if spent + cost > self.budget:
                continue
            ok, reason = muts.validate(self.net, cand, cfg, self.cooldown)
            if not ok:
                continue
            applied, detail = muts.apply(self.net, cand)
            if not applied:
                continue
            spent += cost
            chosen.append({**cand, "accepted": True, "reason": reason, "budget_cost": cost, "detail": detail})
        return chosen

    def _greedy_policy(self) -> List[dict]:
        """Baseline: add between most congested pair; prune idlest. [ASSUMPTION]"""
        from plens import mutations as muts

        cfg = self.cfg
        net = self.net
        chosen = []
        best = None
        best_c = -1.0
        for i in net.alive_nodes():
            for j in net.alive_nodes():
                if i >= j or net.G.has_edge(i, j):
                    continue
                c = net.nodes[i].utilization + net.nodes[j].utilization
                if c > best_c:
                    best_c = c
                    best = {"op": "ADD", "i": i, "j": j, "score": c}
        if best:
            ok, reason = muts.validate(net, best, cfg, self.cooldown)
            if ok:
                applied, detail = muts.apply(net, best)
                if applied:
                    chosen.append({**best, "reason": reason, "budget_cost": cfg.mutation_cost["ADD"], "detail": detail})
        idle = None
        idle_u = 1e18
        for (u, v), e in net.edges.items():
            if not e.failed and e.utilization < idle_u:
                idle_u = e.utilization
                idle = {"op": "PRUNE", "i": u, "j": v, "score": -e.utilization}
        if idle and len(chosen) < cfg.max_mutations_per_step:
            ok, reason = muts.validate(net, idle, cfg, self.cooldown)
            if ok:
                applied, detail = muts.apply(net, idle)
                if applied:
                    chosen.append({**idle, "reason": reason, "budget_cost": cfg.mutation_cost["PRUNE"], "detail": detail})
        return chosen
