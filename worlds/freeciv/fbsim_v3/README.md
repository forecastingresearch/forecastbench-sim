# FreeCiv v3: the ICLR 2027 run

The FreeCiv world of ForecastBench-Sim as it was drawn, elicited and scored for the paper (Jaeho,
September 2026). Eight anchor games played to turn 60 by the game's own AI, 1,000 replays each from
turn 60, and question sets whose truth is the frequency or distribution over all 1,000 replays.
The older `worlds/freeciv/freeciv_world` package (CivBench) is a different pipeline and is not used here.

| folder | contents |
|---|---|
| `run/` | the elicitation harness (`elicit_v1.py`, one question per prompt, as run on 9 September) and the scorer (`score_v1.py`, self-test in `score_selftest.py`); `models_v1.csv` (24 models: reasoning mode, effort or budget, output cap, provider pin, list prices); `ELICITATION.md` (what was run, settings, costs, findings); `PREFLIGHT_prompts.md` (the exact request body and message texts per arm); cost, smoke, reparse and results-table utilities; `full_run.sh` / `launch.sh` / `smoke_run.sh` |
| `draw/` | question generation and QC: candidate pools (`bank_v1.py`), the structural draw (`draw_v1.py`), resolution criteria (`criteria_v1.py`), natural-conditional blocks, report regeneration, corpus consolidation and QC |
| `sets/draw_v1/` | the question sets of draw v1.8 with truth: `bank_750`, `tails_300`, `mirrors_50`, `continuous_300` (1,000 truth values per item), `natcond_600` (400 turn-2 cells) and `natcond_extra_turn1` (124 questions), with `README.md`, `COMPOSITION.md`, `QC_SAMPLES.md`, `report.html` and the continuous normalisation constants |
| `reports/` | the eight turn-60 world reports the models read (`seed*/turn_060_report.txt`) with each anchor's game data |
| `corpus_docs/` | the replay run's log, readiness and substitution notes, family specs and corpus QC |
| `fleet/` | the RunPod scripts of the replay run (2--3 September; ledger and keys are not committed) |
| `results/run1_2026-09-09/` | run 1: `results_v1/` (per-model tables), `scores_v1/` (`SCORES.md`, `score_summary.csv`, `score_items.csv.gz` with 46,176 per-item rows), aggregates behind the figures, raw rows for one model (the other 23 are in the Drive archive), the capability file |
| `paper/` | the generators that deliver into the paper checkout (see below) |

Run 1 is provisional: one question per prompt, while Micropolis batched 50. Run 2 (batched, 50 per
prompt) goes to `results/run2_.../` with its own harness version.

## Delivering into the paper

The paper source is the Overleaf project mirrored at `jaeholee-brown/forecastbench-sim-iclr`. The
generators read this folder and write only FreeCiv-owned files there: the FreeCiv tables, the FreeCiv
figures, the FreeCiv cells of the shared tables, and the backing files under `data/freeciv/`.

```sh
export FBSIM_PAPER_ROOT=/path/to/forecastbench-sim-iclr     # the folder that holds main.tex
export FBSIM_RUN=run1_2026-09-09                             # default
uv run python worlds/freeciv/fbsim_v3/paper/deliver_backing.py
uv run python worlds/freeciv/fbsim_v3/paper/make_freeciv_tables.py
uv run python worlds/freeciv/fbsim_v3/paper/make_freeciv_difficulty_table.py
uv run python worlds/freeciv/fbsim_v3/paper/make_freeciv_family_table.py
uv run python worlds/freeciv/fbsim_v3/paper/make_freeciv_figs.py
uv run python worlds/freeciv/fbsim_v3/paper/update_shared_tables.py --check   # then without --check
```

`update_shared_tables.py` edits the FreeCiv cells of `hosting_cost_table.tex`, `roster_table.tex`,
`cost_per_item.tex` and the three validation tables in place, the way Fabio's `analyze_paper.py` and
Nick's `build_metadata.py` edit theirs, so the three compose in any order. `--final` drops the
"(provisional)" label for run 2.

Needs pandas, numpy, scipy and matplotlib. The API key for elicitation is read from
`OPENROUTER_API_KEY` or a `.env` outside this repository; nothing under this folder holds a credential.
