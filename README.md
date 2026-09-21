# P-LENS — Plasticity-pressure-driven Law Enforcement of Network Structure

**Evolving-topology network simulation with ML-guided, budgeted topology mutations
(ADD / PRUNE / REWIRE).** A GRU forecasts traffic stress from sliding windows of
flow features (CIC-IDS2017 or UNSW-NB15); a controller converts per-node pressure,
debt, credit and a global budget into constraint-aware topology mutations, and
evaluates each mutation's delayed counterfactual gain.

> Status: research/demo code. Every formula is a documented **[ASSUMPTION]**, not a
> claim from an original paper; the architecture tags (`[DIAGRAM]`, `[SIM]`, `[ML]`)
> mark which part of the architecture diagram / simulation / ML pipeline a piece of
> code implements.

## Architecture (one simulation step)

```
 failures/recovery → route & load → monitor D,F,C,R → plasticity pressure PP
       → plasticity debt PD → budget gate B → mutation field MF → selector
       → G_(t+1) → delayed counterfactual eval at t+k → mutation credit MC
       → feedback: PD(t+1), B(t+1)                     (ML forecast blends into B's load target)
```

Implemented in `plens/controller.py` (`PLENSController.step`), with one module per
component under `plens/`.

## Layout

| Path | Purpose |
|---|---|
| `network/` | Evolving graph, failures/recovery, M/M/1-style routing & metrics |
| `plens/` | Indicators, pressure, debt, budget, mutation field, selector, delayed eval, credit |
| `ml/` | CIC/UNSW streaming loaders, GRU (`StressGRU`), training, frozen inference, synthetic generator |
| `experiments/` | `Config`, multi-seed runner with common random numbers, Wilcoxon stats, ablations |
| `visualization/` | Timeseries, policy/ablation bars, topology evolution, training curves |
| `tests/` | 79 pytest tests covering all components |
| `main.py` | End-to-end pipeline entry point |

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows; Linux: .venv/bin/pip
.venv/Scripts/python -m pytest tests            # 79 tests, ~7 s

# Full demo without any external data (~5 min):
python main.py --dataset synthetic

# Real CIC-IDS2017 (downloads/needs DataSets/, see below), trains the GRU first:
python main.py --dataset cic

# Fast smoke test:
python main.py --fast --skip-ablations
```

All artifacts land in `results/`: `experiment_summary.json`, `ablation_summary.json`,
`ml_metrics.json`, per-run JSONs, and PNG figures.

## Datasets

Place the CSVs under the path in `experiments/config.py`
(`DEFAULT_DATASET_ROOT`), or pass a different `dataset_root`:

- **CIC-IDS2017** `MachineLearningCVE`: the 8 weekday CSVs. Weeks are used as
  time: Mon+Tue train, Wed val, Thu test, Fri simulation. Files have no
  Timestamp/IP — file order is time. Attack labels are **never** model inputs;
  they only feed window metadata.
- **UNSW-NB15**: train/test CSVs; chronological split by fraction.
- **synthetic**: no files needed — diurnal Poisson background + lognormal flow
  sizes + port-scan bursts (`ml/synthetic.py`).

The dataset supplies traffic/condition information only; the topology itself is
simulated, never taken from data.

## The P-LENS loop

1. **Monitor** — per-node indicators on the *usable* topology: drift `D`,
   traffic mismatch `F`, congestion `C`, resilience deficiency `R`.
2. **Plasticity pressure** `PP = EMA(w·[D,F,C,R])`.
3. **Plasticity debt** `PD(t+1) = γ·PD + α·PP − relief` (relief after successful
   mutations, endpoints + 1-hop neighbours).
4. **Budget** `B(t+1)` — refill from mean debt and delayed gains, penalized by
   stability cost; hard cap `B_max`, per-step cap `max_mutations_per_step`.
5. **Mutation field** `MF_ij^op = gate(B) · (w_need·PP + w_debt·PD + w_ben·benefit
   + w_cred·MC − w_cost·cost − w_pen·penalty)`.
6. **Selector** — greedy by score under hard constraints: degree caps, min
   degree, connectivity required, λ-target on usable average degree, cooldown.
   Every rejection is logged with its reason.
7. **Delayed evaluation at t+k** — counterfactual shadow: re-route identical
   demand on the topology with the mutation reverted; gain
   `G = w_u·utility + w_r·resilience − w_s·stability(churn)`.
8. **Credit** `MC_ij ← λ·MC + (1−λ)·tanh(G)` feeds the next mutation field.

### ML role (deliberately narrow)

The GRU predicts Low/Medium/High stress for the next window from `L=12` scaled
aggregate-flow features. The label only **modulates the offered-load intensity
target** for the next step (`stress_scales` × `ml_blend_beta`); it never
directly chooses ADD/PRUNE/REWIRE. Ablating ML (`no_ml`) therefore changes
results only through the demand path — verified by `tests/test_ml_integration.py`.

## Baselines & ablations

- Baselines (same constraints, same per-step caps, same CRN seeds): `static`
  (no mutations), `random` (score-blind order, dedicated RNG stream),
  `greedy` (add between most congested pair, prune idlest edge).
- Ablations, each isolating exactly one mechanism while keeping the full
  P-LENS machinery: `no_ml`, `no_debt`, `no_credit`, `no_budget`.
- Stats: mean ± 95% CI over seeds, paired Wilcoxon signed-rank vs P-LENS.

## Results snapshot (demo runs, 3 seeds × 120 steps)

| Scenario | Best policy on latency | Notes |
|---|---|---|
| CIC-IDS2017 sim segment | **P-LENS** 1.80 ms vs static 3.84 / greedy 2.03 / random 1.74 | random edges out P-LENS on latency but mutates ~6× more (470 vs 81) and loses 0.023 more throughput; `no_debt` collapses to 3.28 ms (3 mutations), `no_budget` reaches 1.44 ms by mutating at max rate |
| Synthetic | **P-LENS** 1.87 ms vs static 2.95 / greedy 2.13 / random 1.58 | random competitive but mutates 5× more; `no_budget` lowers latency further by mutating at max rate (budget trades latency for stability/churn) |

Interpretation cautions: few seeds (CI bands overlap), counterfactual eval is a
simulation-only oracle, and per-flow share model is a bottleneck proxy — see the
[ASSUMPTION] tags in code for the full list.

## Configuration

Everything lives in `experiments/config.py::Config` (dataclass, one line of
comment per parameter): topology type/size, traffic intensity, failure rates,
all P-LENS weights/costs/limits, ML hyperparameters, dataset paths.
