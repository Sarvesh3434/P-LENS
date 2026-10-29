"""P-LENS controller: pressure → debt → budget → interaction-aware selection →
matched counterfactual evaluation → credit.

The loop is intentionally explicit so every mechanism can be ablated:

  failures/recovery → route & load → monitor D,F,C,R → pressure PP
  → debt PD (duration/severity/criticality) → budget B (dual ascent)
  → mutation field MF → interaction-aware selector → G_{t+1}
  → replay counterfactual at t+k (matched demand/failures) → credit MC
  → feedback PD(t+1), B(t+1), and the ML forecast's load/budget signal.

Legacy shadow evaluation is retained for comparison; the default is the
matched replay estimator in :mod:`plens.evaluation`.
"""

from __future__ import annotations

import copy
import time
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from experiments.config import Config
from network.failures import apply_failures
from network.graph import EvolvingNetwork, canon
from network.traffic import route_and_load, scale_demand
from plens.budget import update_budget
from plens.credit import update_credit
from plens.debt import (
    DebtState,
    criticality_from_exposure,
    failure_penalty,
    relief_for_mutation,
    update_debt,
)
from plens.evaluation import (
    credit_is_valid,
    evaluate_pending,
    evaluate_replay,
)
from plens.indicators import compute_indicators
from plens import mutations as muts
from plens.mutation_field import expected_benefit, generate_field
from plens.pressure import plasticity_pressure
from plens.selector import select_mutations


class PLENSController:
    def __init__(self, cfg: Config, net: EvolvingNetwork, ml_predictor=None, base_rng=None):
        self.cfg = cfg
        self.net = net
        self.ml = ml_predictor
        n = cfg.network_size
        self.pp = np.zeros(n)
        self.debt = DebtState.zeros(n)
        self.pd = self.debt.pd
        self.budget = cfg.mutation_budget_initial
        self.mc: Dict[tuple, float] = {}
        self.cooldown: Dict[tuple, int] = {}
        self.pending: List[dict] = []
        self.history: List[dict] = []
        self.rejection_log: List[dict] = []
        self.topology_snapshots: List[list] = [[list(e) for e in net.G.edges()]]
        self.rng = base_rng if base_rng is not None else np.random.default_rng(cfg.seed)
        self.step_log: Dict[int, dict] = {}
        self._token = 0
        self._bandit_counts = {"ADD": 0, "PRUNE": 0, "REWIRE": 0}
        self._bandit_values = {"ADD": 0.0, "PRUNE": 0.0, "REWIRE": 0.0}
        self._overload_hist = deque(maxlen=max(cfg.budget_overload_window, 1))
        self._eval_hist = deque(maxlen=max(cfg.budget_overload_window, 1))
        self.ml_budget_factor: float | None = None
        self._ablate()

    def _ablate(self) -> None:
        a = {x.strip() for x in str(self.cfg.ablation).split(",") if x.strip()}
        self.use_ml = self.ml is not None and "no_ml" not in a
        self.use_debt = "no_debt" not in a
        self.use_credit = "no_credit" not in a and "no_evaluation" not in a
        self.use_budget = "no_budget" not in a
        self.use_uncertainty = "no_uncertainty" not in a
        self.use_interaction = "no_interaction" not in a
        self.fixed_score = "fixed_score" in a
        self.do_eval = "no_evaluation" not in a
        self.eval_mode = self.cfg.eval_mode
        if "no_counterfactual" in a:
            self.eval_mode = "before_after"
        self.budget_mode = self.cfg.budget_mode
        if self.cfg.policy in ("fixed_budget", "threshold"):
            self.budget_mode = "fixed"
        elif self.cfg.policy == "adaptive_budget":
            self.budget_mode = "legacy"
        if not self.use_budget:
            self.budget = self.cfg.mutation_budget_max

    # ------------------------------------------------------------------ step
    def step(
        self,
        t: int,
        demand: np.ndarray,
        feature_window_hist: Optional[np.ndarray] = None,
        intensity: float | None = None,
    ) -> dict:
        cfg = self.cfg
        net = self.net
        net.step = t
        t0_clock = time.perf_counter()

        # Replay evaluation needs the complete pre-failure world at this step.
        pre_snapshot = None
        rng_state = copy.deepcopy(net.rng.bit_generator.state)
        if self.eval_mode == "replay_counterfactual" and self.do_eval:
            pre_snapshot = copy.deepcopy(net)

        apply_failures(net)

        # --- [ML] forecast: continuous attack probability + uncertainty ---
        ml_pred = None
        base_intensity = float(cfg.traffic_intensity if intensity is None else intensity)
        target_intensity = base_intensity
        self.ml_budget_factor = None
        if self.use_ml and feature_window_hist is not None and t >= cfg.window_length_L:
            ml_pred = self.ml.predict_stress(feature_window_hist)
            prior = float(cfg.ml_prior_attack)
            p = ml_pred.get("attack_prob")
            if p is None:
                p = {"Low": 0.05, "Medium": 0.50, "High": 0.90}.get(str(ml_pred.get("label", "")), prior)
            p = float(p)
            u = float(ml_pred.get("uncertainty", 0.0))
            denom = max(prior, 1.0 - prior)
            delta = (p - prior) / denom
            target_intensity = base_intensity * (1.0 + cfg.ml_load_gain * delta)
            self.ml_budget_factor = (1.0 + cfg.ml_budget_gain * delta) * (
                1.0 - cfg.ml_uncertainty_damping * u
            )
            ml_pred = {**ml_pred, "load_factor": target_intensity / max(base_intensity, 1e-9)}

        target_intensity = float(np.clip(target_intensity, 0.20, 1.20))
        net.set_demand(scale_demand(demand, net, target_intensity))
        metrics = route_and_load(net)

        ind = compute_indicators(net, net.demand)
        self.pp = plasticity_pressure(ind, cfg, self.pp)
        criticality = criticality_from_exposure(net, cfg.criticality_floor)
        raw_unresolved = float(
            np.mean(np.clip((self.pp - cfg.debt_tolerance) / max(1.0 - cfg.debt_tolerance, 1e-9), 0.0, 1.0))
        )
        self._overload_hist.append(raw_unresolved)
        overload = float(np.mean(self._overload_hist)) if self._overload_hist else 0.0

        for k in list(self.cooldown):
            self.cooldown[k] -= 1
            if self.cooldown[k] <= 0:
                del self.cooldown[k]

        control_t0 = time.perf_counter()
        mutations: List[dict] = []
        rejected: List[dict] = []
        if cfg.policy == "static":
            mutations = []
        elif cfg.policy == "random":
            mutations = self._random_policy()
        elif cfg.policy == "greedy":
            mutations = self._greedy_policy()
        elif cfg.policy == "threshold":
            mutations, rejected = self._threshold_policy()
        elif cfg.policy == "optimization":
            mutations, rejected = self._optimization_policy()
        elif cfg.policy == "bandit":
            mutations, rejected = self._bandit_policy()
        else:
            mutations, rejected = self._plens_policy()

        spent = sum(float(m.get("budget_cost", 0.0)) for m in mutations)
        control_seconds = time.perf_counter() - control_t0

        mutation_records: List[dict] = []
        logged: List[dict] = []
        for m in mutations:
            detail = m.get("detail") or {}
            detail["before"] = dict(net.last_metrics)
            if detail.get("added"):
                self.cooldown[canon(*detail["added"])] = cfg.cooldown_steps
            if detail.get("removed"):
                self.cooldown[canon(*detail["removed"])] = cfg.cooldown_steps
            self._token += 1
            token = self._token
            rec_m = {
                "timestep": t,
                "mutation": m["op"],
                "source_node": m["i"],
                "target_node": m.get("l", m["j"]),
                "removed_edge": detail.get("removed"),
                "added_edge": detail.get("added"),
                "score": m.get("score"),
                "interaction_bonus": m.get("interaction_bonus", 0.0),
                "budget_cost": m.get("budget_cost"),
                "reason": m.get("reason"),
                "delayed_eval": None,
            }
            mutation_records.append(rec_m)
            logged.append({"cand": {k: v for k, v in m.items() if k != "detail"}, "token": token})
            if self.do_eval:
                delay = 1 if self.eval_mode == "immediate" else cfg.delay_k
                self.pending.append(
                    {
                        "t0": t,
                        "t_eval": t + delay,
                        "token": token,
                        "mutation": m,
                        "record": rec_m,
                        "detail": detail,
                        "n_mutations": len(mutations),
                    }
                )

        self.step_log[t] = {
            "pre": pre_snapshot,
            "rng_state": rng_state,
            "demand": net.demand.copy(),
            "mutations": logged,
        }

        # --- delayed / matched counterfactual evaluation [F2] ---
        eval_t0 = time.perf_counter()
        gains: List[float] = []
        stab_costs: List[float] = []
        relief = np.zeros(cfg.network_size)
        failure = np.zeros(cfg.network_size)
        still: List[dict] = []
        n_success = 0
        n_fail = 0
        confidences: List[float] = []
        for item in self.pending:
            if item["t_eval"] > t:
                still.append(item)
                continue
            ev = self._evaluate_item(item)
            gain_credit = ev.get("gain_lcb", ev["G"]) if self.use_uncertainty else ev["G"]
            confidence = ev.get("confidence", 1.0) if self.use_uncertainty else 1.0
            confidences.append(float(confidence))
            success = float(gain_credit) > 0.0
            n_success += int(success)
            n_fail += int(not success)
            self._eval_hist.append(1.0 if success else 0.0)
            gains.append(float(gain_credit))
            stab_costs.append(float(ev.get("stability_cost", 0.0)))

            detail = item["detail"]
            endpoints: List[int] = []
            if detail.get("added"):
                endpoints += list(detail["added"])
            if detail.get("removed"):
                endpoints += list(detail["removed"])
            touched = set(endpoints)
            for node in list(touched):
                touched |= net.neighbours(node)
            if self.use_debt:
                relief += relief_for_mutation(
                    cfg.network_size,
                    list(touched),
                    cfg,
                    success,
                    gain=gain_credit,
                    confidence=confidence,
                )
                failure += failure_penalty(cfg.network_size, list(touched), cfg, success)

            valid_credit = (
                credit_is_valid(ev, cfg) if self.use_uncertainty else abs(float(gain_credit)) >= cfg.cf_min_effect
            )
            if self.use_credit and valid_credit:
                if detail.get("added"):
                    update_credit(self.mc, *detail["added"], gain_credit, cfg.credit_lambda)
                if detail.get("removed"):
                    update_credit(self.mc, *detail["removed"], gain_credit, cfg.credit_lambda)
            if cfg.policy == "bandit":
                self._update_bandit(item["mutation"]["op"], float(gain_credit))

            m = item["mutation"]
            m["delayed_eval"] = ev
            item["record"]["delayed_eval"] = ev
            item["record"]["gain_lcb"] = float(ev.get("gain_lcb", ev["G"]))
            item["record"]["confidence"] = float(ev.get("confidence", 1.0))
        self.pending = still
        relief = np.clip(relief, 0.0, cfg.debt_relief)
        failure = np.clip(failure, 0.0, cfg.debt_failure_penalty * max(len(mutations), 1))
        eval_seconds = time.perf_counter() - eval_t0

        # --- feedback: PD(t+1) [F3], B(t+1) [F4] ---
        self.debt = update_debt(
            self.debt,
            self.pp,
            relief,
            criticality,
            cfg,
            enabled=self.use_debt,
            failure=failure,
        )
        self.pd = self.debt.pd

        mean_gain = float(np.mean([max(g, 0.0) for g in gains])) if gains else 0.0
        gain_std = float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0
        mean_stab = float(np.mean(stab_costs)) if stab_costs else 0.0
        failure_rate = float(np.mean(self._eval_hist)) if self._eval_hist else 0.0
        eval_uncertainty = 1.0 - float(np.mean(confidences)) if confidences else 0.0
        self.budget = update_budget(
            self.budget,
            spent,
            float(np.mean(self.pd)),
            mean_gain,
            mean_stab,
            cfg,
            enabled=self.use_budget,
            gain_std=gain_std,
            overload=overload,
            failure_rate=failure_rate,
            uncertainty=eval_uncertainty,
            mode=self.budget_mode,
        )
        if self.use_ml and self.ml_budget_factor is not None and self.use_budget:
            self.budget = float(
                np.clip(
                    self.budget * self.ml_budget_factor,
                    cfg.mutation_budget_min,
                    cfg.mutation_budget_max,
                )
            )

        self.topology_snapshots.append([list(e) for e in net.G.edges()])
        rec = {
            "t": t,
            "metrics": metrics,
            "pp_mean": float(self.pp.mean()),
            "pd_mean": float(self.pd.mean()),
            "pd_max": float(self.pd.max()),
            "debt_persistence_mean": float(self.debt.persistence.mean()),
            "debt_duration_mean": float(self.debt.duration.mean()),
            "criticality_mean": float(criticality.mean()),
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
            "n_success": n_success,
            "n_fail": n_fail,
            "mc_mean": float(np.mean(list(self.mc.values())) if self.mc else 0.0),
            "indicators_mean": {k: float(v.mean()) for k, v in ind.items()},
            "overload": overload,
            "eval_uncertainty": eval_uncertainty,
            "control_seconds": control_seconds,
            "eval_seconds": eval_seconds,
            "step_seconds": time.perf_counter() - t0_clock,
        }
        self.history.append(rec)
        if rejected:
            self.rejection_log.append({"t": t, "rejected": rec["rejections"]})
        self._trim_logs(t)
        return rec

    # ------------------------------------------------------------- policies
    def _field(self, use_credit: bool | None = None):
        cfg = self.cfg
        field = generate_field(
            self.net,
            self.pp,
            self.pd if self.use_debt else np.zeros_like(self.pd),
            self.mc,
            self.budget,
            cfg,
            use_credit=self.use_credit if use_credit is None else use_credit,
        )
        if self.fixed_score:
            for c in field:
                c["score"] = expected_benefit(self.net, int(c["i"]), int(c.get("l", c["j"])), c["op"])
        return field

    def _plens_policy(self):
        field = self._field()
        weight = self.cfg.interaction_weight if self.use_interaction else 0.0
        return select_mutations(
            self.net, field, self.budget, self.cfg, self.cooldown, interaction_weight=weight
        )

    def _threshold_policy(self):
        if float(self.pp.max()) < self.cfg.threshold_trigger:
            return [], []
        field = self._field(use_credit=False)
        mutations, rejected = select_mutations(
            self.net, field, self.budget, self.cfg, self.cooldown, interaction_weight=0.0
        )
        return mutations, rejected

    def _optimization_policy(self):
        """Greedy knapsack over expected benefit/cost (no debt or credit)."""
        cfg = self.cfg
        net = self.net
        field = generate_field(
            net, self.pp, np.zeros_like(self.pd), {}, self.budget, cfg, use_credit=False
        )
        for c in field:
            c["score"] = expected_benefit(net, int(c["i"]), int(c.get("l", c["j"])), c["op"]) / max(
                cfg.mutation_cost[c["op"]], 1e-9
            )
        field.sort(key=lambda c: c["score"], reverse=True)
        chosen, rejected = [], []
        spent = 0.0
        used = set()
        for c in field:
            if len(chosen) >= cfg.max_mutations_per_step:
                break
            cost = cfg.mutation_cost[c["op"]]
            if spent + cost > self.budget + 1e-9:
                rejected.append({**c, "reason": "insufficient_budget"})
                continue
            ok, reason = __import__("plens.mutations", fromlist=["validate"]).validate(
                net, c, cfg, self.cooldown
            )
            if not ok:
                rejected.append({**c, "reason": reason})
                continue
            applied, detail = muts.apply(net, c)
            if not applied:
                rejected.append({**c, "reason": "apply_failed"})
                continue
            key = canon(c["i"], c.get("l", c["j"]))
            if key in used:
                muts.revert_one(net, detail)
                rejected.append({**c, "reason": "edge_already_used_this_step"})
                continue
            used.add(key)
            if c["op"] == "REWIRE":
                used.add(canon(c["i"], c["j"]))
            spent += cost
            chosen.append({**c, "accepted": True, "reason": reason, "budget_cost": cost, "detail": detail})
        return chosen, rejected

    def _bandit_policy(self):
        """UCB over operation types; a lightweight learning-based baseline."""
        field = self._field(use_credit=False)
        if not field:
            return [], []
        total = sum(self._bandit_counts.values()) + 1
        best_op = None
        best_ucb = -1e18
        for op in ("ADD", "PRUNE", "REWIRE"):
            if not any(c["op"] == op for c in field):
                continue
            n = self._bandit_counts[op]
            ucb = self._bandit_values[op] + self.cfg.bandit_ucb_c * np.sqrt(np.log(total) / (n + 1))
            if ucb > best_ucb:
                best_ucb = ucb
                best_op = op
        candidates = [c for c in field if c["op"] == best_op]
        candidates.sort(key=lambda c: c["score"], reverse=True)
        chosen, rejected = [], []
        spent = 0.0
        for c in candidates:
            if len(chosen) >= self.cfg.max_mutations_per_step:
                break
            cost = self.cfg.mutation_cost[c["op"]]
            if spent + cost > self.budget + 1e-9:
                continue
            from plens import mutations as muts

            ok, reason = muts.validate(self.net, c, self.cfg, self.cooldown)
            if not ok:
                rejected.append({**c, "reason": reason})
                continue
            applied, detail = muts.apply(self.net, c)
            if applied:
                spent += cost
                chosen.append({**c, "accepted": True, "reason": reason, "budget_cost": cost, "detail": detail})
        return chosen, rejected

    def _update_bandit(self, op: str, reward: float) -> None:
        self._bandit_counts[op] = self._bandit_counts.get(op, 0) + 1
        n = self._bandit_counts[op]
        old = self._bandit_values.get(op, 0.0)
        self._bandit_values[op] = old + (reward - old) / n

    def _random_policy(self) -> List[dict]:
        """Baseline: random-adaptive under the SAME hard constraints and max count."""
        cfg = self.cfg
        field = generate_field(
            self.net, self.pp, self.pd, self.mc, self.budget, cfg, use_credit=False
        )
        if not hasattr(self, "_base_rng"):
            self._base_rng = np.random.default_rng(cfg.seed)
        self._base_rng.shuffle(field)
        chosen = []
        spent = 0.0
        from plens import mutations as muts

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
        """Baseline: add between most congested pair; prune idlest."""
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

    # ----------------------------------------------------------- evaluation
    def _evaluate_item(self, item: dict) -> Dict[str, float]:
        cfg = self.cfg
        if self.eval_mode == "replay_counterfactual":
            t0, t_eval = item["t0"], item["t_eval"]
            logs = []
            for s in range(t0, t_eval):
                logs.append(self.step_log[s])
            final = dict(self.step_log.get(t_eval, self.step_log[t0]))
            final["mutations"] = []
            logs.append(final)
            snapshot = self.step_log[t0]["pre"]
            if snapshot is None:
                return evaluate_pending(self.net, item["detail"], cfg, item["n_mutations"])
            return evaluate_replay(
                snapshot,
                logs,
                horizon=len(logs),
                token=item["token"],
                cfg=cfg,
                n_mutations=1,
            )
        if self.eval_mode == "immediate":
            return evaluate_pending(self.net, item["detail"], cfg, item["n_mutations"])
        return evaluate_pending(self.net, item["detail"], cfg, item["n_mutations"])

    def _trim_logs(self, t: int) -> None:
        keep_from = min((p["t0"] for p in self.pending), default=t)
        keep_from = min(keep_from, t) - 2
        for k in list(self.step_log):
            if k < keep_from:
                del self.step_log[k]

