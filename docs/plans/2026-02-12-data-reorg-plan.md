# Data Directory Reorganization — Implementation Plan

Design: [2026-02-12-data-reorg-design.md](./2026-02-12-data-reorg-design.md)

## Tasks

### 1. Delete dead artifacts

Delete the following unreferenced files and directories:

- `data/questions_single_game/` (empty dir)
- `data/games/seed6_partial_backup_data.json`
- `data/questions/seed6_partial_backup/`
- `data/questions_conditional_only/` (seed0-only precursor, superseded)
- `data/evaluations/seed0_*.json` (6 files — early single-seed experiments)
- `data/evaluations/eval_42_*.json` (4 files — old Dec 2025 runs)

Expected: 12 fewer files/dirs, no script references broken.

### 2. Move conditional question sets into nested structure

```
mkdir -p data/conditional/republic data/conditional/gold500

mv data/questions_baseline_eval          data/conditional/republic/baseline
mv data/questions_conditional_eval       data/conditional/republic/conditional
mv data/questions_conditional_no_eval    data/conditional/republic/conditional_no
mv data/questions_given_that_eval        data/conditional/republic/given_that
mv data/questions_post_intervention_eval data/conditional/republic/post_intervention
mv data/questions_gold500_eval           data/conditional/gold500/conditional
mv data/questions_gold500_post_intervention_eval data/conditional/gold500/post_intervention
mv data/questions_null_conditional       data/conditional/null_conditional
```

Expected: 8 top-level `questions_*` dirs replaced by single `conditional/` tree. Internal contents unchanged.

### 3. Reorganize evaluations directory

```
mkdir -p data/evaluations/results data/evaluations/plots data/evaluations/runs

mv data/evaluations/*.png data/evaluations/*.pdf   data/evaluations/plots/
mv data/evaluations/parallel_eval_*                 data/evaluations/runs/
mv data/evaluations/*.json                          data/evaluations/results/
```

Expected: `data/evaluations/` contains only `results/`, `plots/`, `runs/` subdirs.

### 4. Update setup scripts (output paths)

Files:
- `scripts/setup_republic_conditional_eval.py` — change 5 output dir paths + print statements
- `scripts/setup_gold500_conditional_eval.py` — change output dir + print statement
- `scripts/setup_gold500_post_intervention_eval.py` — change output dir + print statement

Path mapping:
- `data/questions_baseline_eval` → `data/conditional/republic/baseline`
- `data/questions_conditional_eval` → `data/conditional/republic/conditional`
- `data/questions_conditional_no_eval` → `data/conditional/republic/conditional_no`
- `data/questions_given_that_eval` → `data/conditional/republic/given_that`
- `data/questions_post_intervention_eval` → `data/conditional/republic/post_intervention`
- `data/questions_gold500_eval` → `data/conditional/gold500/conditional`
- `data/questions_gold500_post_intervention_eval` → `data/conditional/gold500/post_intervention`

Expected: Scripts generate into new locations. No behavioral change.

### 5. Update evaluation/analysis scripts (input paths)

Files:
- `scripts/evaluate_llm_forecasts_parallel.py` — default output `data/evaluations/` → `data/evaluations/runs/`
- `scripts/analyze_optimal_brier.py` — hardcoded eval paths + source_glob paths
- `scripts/compute_difficulty_scores.py` — default eval dir
- `scripts/validate_difficulty_calibration.py` — default eval dir, plot dir already correct
- `scripts/plot_conditional_framing_comparison.py` — hardcoded eval + plot output paths

Expected: All analysis scripts read from / write to new paths.

### 6. Update null conditional generation script + shell script

Files:
- `scripts/generate_null_conditional_questions.py` — default `--output-dir` + docstring
- `run_null_conditional_eval.sh` — data dir references

Expected: Null conditional pipeline uses new path.

### 7. Update documentation references

Files:
- `docs/experiments/null_conditional_framing.md`
- `docs/notes/evaluation_setup.md`

Expected: Docs reference correct paths.

### 8. Write README files

Create three documentation files:

**`data/README.md`** — Top-level overview:
- Pipeline stages (game execution → serialization → questions → conditional → evaluation)
- What each directory contains
- Key scripts for each stage

**`data/conditional/README.md`** — Conditional experiment design:
- Intervention types (republic, gold500)
- Framing variants and their purpose
- How to add a new intervention

**`data/conditional/null_conditional/README.md`** — Framing placebo control:
- What null conditionals test (framing effect independent of intervention)
- How they're generated (negated condition, same ground truth)
- The 114 subdirectories (one per shuffled question set)

Expected: Anyone new to the repo can understand the data layout.

### 9. Verify

- `grep -r "questions_baseline_eval\|questions_conditional_eval\|questions_conditional_no_eval\|questions_given_that_eval\|questions_post_intervention_eval\|questions_gold500_eval\|questions_gold500_post_intervention_eval\|questions_null_conditional\|questions_single_game\|questions_conditional_only" scripts/ src/ docs/ *.sh` returns no matches
- All moved directories contain the same file counts as before
- `uv run python scripts/setup_republic_conditional_eval.py --help` runs without import errors
- `uv run python scripts/evaluate_llm_forecasts_parallel.py --help` runs without import errors

### 10. Update CLAUDE.md

Add a note about the data directory structure and link to `data/README.md`.
