# Implementation Plan: Continuous Question Evaluation at Scale

**Design:** `docs/plans/2026-02-12-continuous-eval-design.md`
**Goal:** Extend the evaluation pipeline to handle continuous questions (percentile estimates, CRPS/MAE scoring) alongside binary questions.

## Architecture

The main eval pipeline (`evaluate_llm_forecasts_parallel.py`) is binary-only. We extend it by:
1. Fixing question loading to preserve numeric ground truth for continuous questions
2. Adding a continuous prompt builder and percentile response parser
3. Implementing CRPS and MAE metrics
4. Splitting batches by question type and dispatching metrics accordingly

**Tech stack:** Python, asyncio, litellm, existing CivBench evaluation infrastructure.

---

## Task 1: Fix question loading for continuous ground truth

**File:** `src/civrealm/evaluation/sampling.py`

### Steps

1. In `_load_from_combined_file()` (line ~78-116):
   - After `resolution = q.get("resolution", {})`, read `question_type = q.get("question_type", "binary")`
   - Replace `"ground_truth": bool(answer)` (line 110) with:
     ```python
     if question_type == "continuous":
         ground_truth = resolution.get("value")
     else:
         ground_truth = bool(resolution.get("answer"))
     ```
   - For continuous, skip questions where `resolution.get("value")` is None
   - Add `"question_type": question_type` to the returned dict

2. In `_load_from_game_directories()` (line ~119-178):
   - Same changes: read `question_type`, branch ground_truth extraction, add to dict
   - Replace `"ground_truth": bool(answer)` (line 172) with same branching logic

3. Verify both loading paths preserve `question_type` and numeric `ground_truth`.

**Expected:** `load_all_questions()` returns dicts with `question_type` and correct `ground_truth` types.

---

## Task 2: Add CRPS and MAE to metrics module

**File:** `src/civrealm/metrics.py`

### Steps

1. Add `compute_crps(percentiles: dict, true_value: float) -> float`:
   - Takes `{"p10": float, "p25": float, "p50": float, "p75": float, "p90": float}` and true value
   - Implements quantile-weighted pinball loss approximation:
     ```
     quantile_levels = [0.10, 0.25, 0.50, 0.75, 0.90]
     CRPS ≈ (2/N) * Σ ρ_τ(true - q_τ)
     where ρ_τ(u) = u*(τ - I(u<0))  (pinball/quantile loss)
     ```
   - Returns non-negative float (0 = perfect)

2. Add `compute_mae(predicted: float, true_value: float) -> float`:
   - Returns `abs(predicted - true_value)`

3. Add `compute_aggregate_crps(all_percentiles: list[dict], all_true_values: list[float]) -> float`:
   - Computes mean CRPS across a list of question results

4. Add `compute_aggregate_mae(all_p50: list[float], all_true_values: list[float]) -> float`:
   - Computes mean MAE across a list of question results

**Expected:** New functions importable from `civrealm.metrics`. CRPS=0 for perfect quantile match, increases with miscalibration.

---

## Task 3: Add continuous prompt builder and percentile parser

**File:** `src/civrealm/evaluation/parallel_evaluator.py`

### Steps

1. Add `build_continuous_batch_prompt(questions: list[dict], world_report: str) -> str`:
   - Same structure as `build_batch_prompt` but with percentile instructions
   - Numbered questions, same world report header
   - Instruction block asks for `<<<PERCENTILES>>>` delimited response
   - Format per question: `Q{i}: p10=X, p25=Y, p50=Z, p75=A, p90=B`

2. Add `parse_batch_percentiles(response: str, num_questions: int) -> list[dict | None]`:
   - Try delimiter extraction first (`<<<PERCENTILES>>>...<<<END>>>`)
   - Within the block, match lines with `Q{n}:` prefix or numbered lines
   - From each line, extract p10 through p90 using regex `p\d+\s*=\s*(-?\d+\.?\d*)`
   - Return `list[dict | None]` where dict has keys p10, p25, p50, p75, p90
   - Fallback: try JSON array of objects, then bare lines with 5 numbers each
   - Pad with None if fewer results than expected

3. Add `ContinuousBatchPredictionResult` dataclass:
   - `model_id: str`
   - `percentiles: list[dict | None]` (one per question)
   - `latency_ms: float`
   - `error: str | None`

4. Add `query_model_continuous_batch_async()`:
   - Same signature as `query_model_batch_async` but calls `parse_batch_percentiles`
   - Returns `ContinuousBatchPredictionResult`

5. Add `evaluate_continuous_question_batch()`:
   - Parallel to `evaluate_question_batch()` but for continuous questions
   - Calls `build_continuous_batch_prompt` and `query_model_continuous_batch_async`
   - Returns results with `percentiles` instead of `probability`
   - Result format: `{"question_id": ..., "ground_truth": 1500, "question_type": "continuous", "predictions": {"model_id": {"percentiles": {...}, "latency_ms": ..., "error": ...}}}`

**Expected:** Continuous batches produce parseable percentile predictions. Tested with sense_check_eval format.

---

## Task 4: Update main evaluation script for mixed question types

**File:** `scripts/evaluate_llm_forecasts_parallel.py`

### Steps

1. After stratified sampling, split questions by type:
   ```python
   binary_questions = [q for q in questions if q.get("question_type", "binary") == "binary"]
   continuous_questions = [q for q in questions if q.get("question_type") == "continuous"]
   ```
   Create separate batches for each type (group by game, chunk by batch_size).

2. Update `run_batch_evaluation` call (or add parallel calls):
   - Run binary batches through existing `_process_single_batch` (unchanged)
   - Run continuous batches through a new `_process_single_continuous_batch` that uses `evaluate_continuous_question_batch`
   - Merge results, tagging each with `question_type`

3. Refactor `compute_metrics()` to dispatch by question type:
   - Separate results into binary and continuous
   - Binary: existing Brier + ECE logic (unchanged)
   - Continuous: collect percentiles + true values, compute mean CRPS + mean MAE
   - Structure: `model_metrics[model_id] = {"binary": {...}, "continuous": {...}}`
   - Also compute per-template breakdowns for each type

4. Update `print_results_summary()`:
   - Show binary section (Brier, ECE) then continuous section (CRPS, MAE)
   - Show per-template breakdowns for each

5. Update metadata and output JSON to include question type distribution

6. Add `--question-type` CLI arg (choices: "all", "binary", "continuous", default: "all"):
   - Filter questions after loading

**Expected:** `python scripts/evaluate_llm_forecasts_parallel.py -n 5` evaluates both binary and continuous questions, showing separate metrics for each.

---

## Task 5: Integration test with dry run

**File:** No new files, uses existing script

### Steps

1. Run `evaluate_llm_forecasts_parallel.py --dry-run -n 2` and verify:
   - Both binary and continuous questions are sampled
   - Template distribution shows continuous templates
   - No errors in loading or batching

2. Run `evaluate_llm_forecasts_parallel.py --models anthropic/claude-sonnet-4-5-20250929 -n 2 --question-type continuous` to verify continuous-only evaluation works end-to-end

3. Run `evaluate_llm_forecasts_parallel.py --models anthropic/claude-sonnet-4-5-20250929 -n 2` to verify mixed evaluation works

4. Verify output JSON has correct structure with both binary and continuous metrics

**Expected:** All runs complete without errors. Output JSON matches documented structure.

---

## Dependencies

- Task 1: none (independent)
- Task 2: none (independent)
- Task 3: none (independent, uses existing patterns)
- Task 4: depends on Tasks 1, 2, 3
- Task 5: depends on Task 4
