# Data Directory Reorganization

## Problem

`data/` has 13 top-level directories, 9 starting with `questions_`. No documentation explains the structure. Dead artifacts mix with production data. The flat naming obscures the logical hierarchy (intervention type > framing variant).

## Design

### Deletions

- `data/questions_single_game/` (empty)
- `data/games/seed6_partial_backup_data.json` (unreferenced, full seed6 exists)
- `data/questions/seed6_partial_backup/` (same)
- `data/questions_conditional_only/` (seed0-only precursor, superseded by eval dirs)
- `data/evaluations/seed0_*.json` (6 files, early single-seed experiments)
- `data/evaluations/eval_42_*.json` (4 files, old Dec 2025 timestamped runs)

### Directory moves

```
OLD                                    → NEW
data/questions_baseline_eval/          → data/conditional/republic/baseline/
data/questions_conditional_eval/       → data/conditional/republic/conditional/
data/questions_conditional_no_eval/    → data/conditional/republic/conditional_no/
data/questions_given_that_eval/        → data/conditional/republic/given_that/
data/questions_post_intervention_eval/ → data/conditional/republic/post_intervention/
data/questions_gold500_eval/           → data/conditional/gold500/conditional/
data/questions_gold500_post_intervention_eval/ → data/conditional/gold500/post_intervention/
data/questions_null_conditional/       → data/conditional/null_conditional/
```

### Evaluations split

```
data/evaluations/*.json (named results) → data/evaluations/results/
data/evaluations/*.{png,pdf}            → data/evaluations/plots/
data/evaluations/parallel_eval_*        → data/evaluations/runs/
```

### Documentation

- `data/README.md` — overview of directory structure, pipeline stages, and what each folder contains
- `data/conditional/README.md` — explains the conditional forecasting experiment design (interventions, framings)
- `data/conditional/null_conditional/README.md` — explains the framing placebo control

### Scripts requiring path updates

1. `scripts/setup_republic_conditional_eval.py` — output dirs for 5 republic framings
2. `scripts/setup_gold500_conditional_eval.py` — output dir for gold500 conditional
3. `scripts/setup_gold500_post_intervention_eval.py` — output dir for gold500 post-intervention
4. `scripts/generate_null_conditional_questions.py` — default output dir
5. `scripts/evaluate_llm_forecasts_parallel.py` — default eval output path
6. `scripts/analyze_optimal_brier.py` — hardcoded eval paths
7. `scripts/compute_difficulty_scores.py` — default eval dir
8. `scripts/validate_difficulty_calibration.py` — default eval/plot dirs
9. `scripts/plot_conditional_framing_comparison.py` — hardcoded eval/plot paths
10. `run_null_conditional_eval.sh` — data dir references
11. `docs/experiments/null_conditional_framing.md` — path references
12. `docs/notes/evaluation_setup.md` — path references

### Final structure

```
data/
├── README.md
├── games/                              # Serialized game snapshots
│   ├── seed{0-10}_data.json
│   ├── seed0forkgoldadd5000p0_data.json
│   └── seed0forkgoldadd500p0_data.json
├── questions/                          # Unconditional forecasting questions
│   ├── questions_all.json
│   ├── seed{0-10}/
│   └── *.json (base_rates, summary, etc.)
├── conditional/                        # Conditional forecasting experiments
│   ├── README.md
│   ├── republic/                       # Republic government intervention
│   │   ├── baseline/                   # Unconditional framing, control answer
│   │   ├── conditional/                # "If switches to Republic" framing
│   │   ├── conditional_no/             # "If does NOT switch" framing
│   │   ├── given_that/                 # "Given that will switch" framing
│   │   └── post_intervention/          # Post-intervention world report
│   ├── gold500/                        # +500 gold intervention
│   │   ├── conditional/                # "If receives +500 gold" framing
│   │   └── post_intervention/          # Post-intervention
│   └── null_conditional/               # Framing placebo control
│       └── README.md
└── evaluations/                        # LLM forecast evaluation results
    ├── results/                        # Named evaluation outputs
    ├── plots/                          # Figures (png, pdf)
    └── runs/                           # Timestamped raw eval runs
```
