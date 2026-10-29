# FreeCiv v3

The FreeCiv pipeline behind the ForecastBench-Sim paper (Jaeho Lee, September 2026): eight anchor games played to
turn 60, 1,000 replays of each from turn 60, question sets whose truth is the frequency or distribution over the
replays, elicitation through OpenRouter and scoring. Copied from the `freeciv-v3` branch at `daca5a15`; sources,
hashes and changes are in `PROVENANCE.json`.

| folder | contents |
|---|---|
| `fleet/` | RunPod scripts for the replay fleet; `make_bundle.sh` assembles what `bringup.sh` sends to each pod |
| `draw/` | corpus consolidation, candidate pools (`bank_v1.py`), the structural draw (`draw_v1.py`), resolution criteria, natural conditionals, report regeneration, QC |
| `run/` | elicitation (`elicit_v1.py` one question per prompt, `elicit_v2.py` batched, `elicit_natcond_v1.py`, `effort_check_v2.py`), scoring (`score_v1.py`, `score_v2.py`, `score_selftest.py`), row rebuild and reparse, selection fractions |
| `runcfg.py` | a run's configuration (`run.env`) for the Python scripts |

## Runs

A run is one configuration (player count, barbarians, fork turn, horizons, rng range, anchor seeds) with its own
folder, `data/freeciv/fbsim_v3/runs/<run>/`. `fleet/1_new_run.sh` creates it and its `run.env`, e.g.

```bash
fleet/1_new_run.sh p2_t60 --players 2 --barbarians DISABLED --seeds 9201-9240
fleet/1_new_run.sh p5_t120 --fork-turn 120 --seeds 9301-9320      # horizons 150..270 by default
```

`1_new_run.sh` also writes `<run>/shell.sh`; `source` it to export `FBSIM_RUN=<run>` (and set the runbook's shorthands
`R`, `S`, `K`, `V3`). `FBSIM_RUN` (or `--run <run>` on the Python scripts) then selects the run everywhere: the fleet scripts keep the
run's ledger, queues and anchors in `<run>/fleet_state/`, name pods `fbsim3-<run>-...`, ship `run.env` to the pods and
archive to `/vol/fbsim_v3/<run>/`; the draw, elicitation and scoring scripts read the fork turn, horizons, player count
and barbarians from `run.env` and default their folders to the run's:

```
runs/<run>/run.env  shell.sh  fleet_state/  raw/  consol_v4/  bank_v1/  sets/draw_v1/  reports/
            run/models_v2.csv  evals/<label>/  scores/<label>/
```

The numbered scripts in `fleet/` run a whole run, one step each. Each takes the run name first, sources the run's
`shell.sh` and reports what it does; the long steps wait with progress and can be stopped and rerun.

| step | script | does |
|---|---|---|
| 1 | `1_new_run.sh RUN --seeds LO-HI [...]` | create the run (refuses a dirty checkout without `--force`) |
| 2 | `2_anchors.sh RUN [SEED...]` | play the anchors on one pod, pull them, print anchor QC |
| 3 | `3_keep_anchors.sh RUN SEED...` | keep the chosen anchors; offers to delete the anchor pod |
| 4 | `4_start_fleet.sh RUN N_WORKERS [--deadman H]` | archive pod, workers, queue, bringup, lanes |
| 5 | `5_finish.sh RUN` | wait for the workers to drain, offer reruns of twice-failed tasks, finish and delete workers |
| 6 | `6_pull_corpus.sh RUN` | pull, QC, consolidate; offers to delete the archive pod |
| 7 | `7_make_sets.sh RUN --models-csv PATH` | bank, sets, criteria, norm constants, reports, model panel |
| 8 | `8_elicit.sh RUN LABEL [--models ...]` | dry run, then (after confirmation) elicitation into `evals/LABEL/` |
| 9 | `9_score.sh RUN LABEL` | score into `scores/LABEL/` |

`fleet/runs` links to the run folders (`scripts/setup_worktree.sh` creates it), so `source runs/<run>/shell.sh` works
from `fleet/` for manual commands.

Without a run the Python scripts keep the paper's settings (5 players, turn 60, horizons 90..210) and need explicit
folders; with them they reproduce the paper's sets, reports and prompts byte for byte. The step-by-step procedure is
in `RUNBOOK.md`.

Other inputs live outside this folder:

- `FBSIM_FREECIV_V3_DATA`: overrides the elicitation folder (`sets/draw_v1/`, `reports/`, `run/models_v*.csv`), e.g.
  `forecastbench-sim-paper/production/freeciv` for the paper's question sets, reports and model panels.
- `OPENROUTER_API_KEY`, `RUNPOD_API_KEY`: exported, or in the repository's `credentials/.env`; never stored here.
  The fleet's ssh keys are `credentials/freeciv/{runpod_key,archive_key}` (set in `fleet/paths.sh`; see `AGENTS.md`).

Launch scripts, run records and the paper's analysis are in `forecastbench-sim-paper` (`production/freeciv`,
`src/fbsim_paper/assets/freeciv`). Raw responses and scores are in the private data archive.
