#!/usr/bin/env python
"""Copy the FreeCiv backing files into the paper checkout's data/freeciv/ and write its README.

    python worlds/freeciv/fbsim_v3/paper/deliver_backing.py --paper-root /path/to/paper

The paper (the Overleaf mirror) carries only what its numbers come from: the scorer's per-model summary,
the run log's per-model table, the leaderboards, the family and reliability aggregates, the model list with
its settings and the constants fixed at the draw.  Per-item rows, raw responses, code and question sets
stay in this repository (worlds/freeciv/fbsim_v3/).
"""
import shutil

from _common import BACKING_FILES, DATA, MODELS_V1, MODELS_V2, RUN, rel

for stale in (MODELS_V1, MODELS_V2):
    if stale not in BACKING_FILES and (DATA / stale.name).exists():
        (DATA / stale.name).unlink()
        print(f"removed    {rel(DATA / stale.name)} (the other run's model file)")

for src in BACKING_FILES:
    dst = DATA / src.name
    if dst.exists() and dst.read_bytes() == src.read_bytes():
        print(f"unchanged  {rel(dst)}")
        continue
    shutil.copy2(src, dst)
    print(f"copied     {rel(src)} -> {rel(dst)}")

README = f"""# FreeCiv: files behind the paper's numbers

Copied from `worlds/freeciv/fbsim_v3/results/{RUN}/` of the `forecastbench-sim` repository
(branch `freeciv-v3`) by `worlds/freeciv/fbsim_v3/paper/deliver_backing.py`. Do not edit here: the
generators in that folder write the FreeCiv tables (`../appendix_tables/freeciv_*.tex`,
`../freeciv_*_families.tex`, `../freeciv_items_per_anchor.tex`), the FreeCiv figures
(`../../figures/fig_freeciv_*.pdf` with a JSON of every plotted number beside each) and the FreeCiv
cells of the shared tables (`../hosting_cost_table.tex`, `../roster_table.tex`,
`../appendix_tables/cost_per_item.tex`, `../validation_table*.tex`).

| file | what |
|---|---|
| `freeciv_results_wide.csv` | one row per model: ECI, every set x horizon aggregate, calls, cost |
| `freeciv_results_table.md` | the run log's per-model table: calls, cost, parse rate |
| `SCORES.md` | leaderboards with item-bootstrap intervals, by set, horizon and block |
| `family_horizon_scores.csv` | model x set x family x horizon aggregates (difficulty table) |
| `reliability_bands.csv` | bank reliability: mean forecast and truth in ten bands of q, per model |
| `model_scores.csv` | the 24 shared models: ECI and ForecastBench overall |
| `models_v1.csv` or `models_v2.csv` | the run's per-model reasoning mode, effort or budget, output cap, provider pin, list prices |
| `continuous_norm_constants.json` | the per-family constants that divide the continuous CRPS, fixed at the draw |
| `COMPOSITION.md` | item counts of the draw by family, horizon, block and anchor game |
| `freeciv_numbers.json`, `freeciv_summary.json`, `freeciv_family_summary.json`, `freeciv_validation_stats.json` | every number the FreeCiv prose and figures quote, with its source and method |

Per-item scores (46,176 rows), raw model responses, the question sets with their replay truth, the
world reports and the elicitation and scoring code are in the repository above, and in the release
of Appendix F.
"""
(DATA / "README.md").write_text(README)
print(f"wrote      {rel(DATA / 'README.md')}")
