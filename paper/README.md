# P-LENS paper (Springer LNCS)

Research paper describing the P-LENS evolving-network simulation, in the
Springer **LNCS** (`llncs.cls`) format.

## Contents

| File | Purpose |
|---|---|
| `main.tex` | The paper (Springer LNCS). |
| `references.bib` | Bibliography. |
| `llncs.cls`, `splncs04.bst` | Official Springer LNCS class + bib style (bundled). |
| `figures/` | Figures copied from `../results/` (`policy_bars`, `ablation_bars`, `timeseries_*`, `topology_evolution`, `training_curves`). |

## Build

With a LaTeX toolchain (TeX Live / MiKTeX) installed, from this folder:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

or simply `latexmk -pdf main`. The output is `main.pdf`.

The class file `llncs.cls` and style `splncs04.bst` are the official files from
the Springer LNCS author kit, vendored here so the paper compiles offline. If you
prefer, delete them and install the `llncs` package from CTAN, or drop in a
fresh copy from
<https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines>.

## Notes

- Author/title/institute are placeholders in `main.tex` — replace them.
- All reported numbers come from the artifacts in `../results/`
  (`ml_metrics.json`, `runs/experiment_summary.json`,
  `ablations/ablation_summary.json`) and the repository `README.md` demo
  snapshot. See `Section 7 (Threats to Validity)` of the paper for caveats.
- The figures are PNG snapshots produced by `visualization/plots.py`; regenerate
  them by re-running `python main.py` at the repository root.
