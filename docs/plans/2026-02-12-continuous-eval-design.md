# Design: Continuous Question Evaluation at Scale

## Goal

Extend the main evaluation pipeline (`evaluate_llm_forecasts_parallel.py`) to handle continuous questions alongside binary questions, enabling large-scale evaluation of all 4 question types: binary unconditional, binary conditional, continuous unconditional, continuous conditional.

## Design Decisions

1. **Separate batches by type**: Binary and continuous questions are split into separate batches before prompting. Each batch gets a type-specific prompt format.
2. **5 percentiles**: Continuous prompts ask for p10, p25, p50, p75, p90.
3. **CRPS + MAE**: Two continuous metrics — CRPS (calibration+sharpness) and MAE (point accuracy).
4. **Results split by type**: Output JSON separates binary and continuous metrics per model.

## Architecture

### Data Flow

```
load_all_questions()  →  stratified_sample()  →  split by question_type
                                                    ↓              ↓
                                              binary batches   continuous batches
                                                    ↓              ↓
                                           binary prompt     continuous prompt
                                           (probability)     (percentiles)
                                                    ↓              ↓
                                           parse probs      parse percentiles
                                                    ↓              ↓
                                                merge results
                                                    ↓
                                           compute_metrics()
                                           (dispatch by type)
                                                    ↓
                                           save + print summary
```

### Loading Changes (sampling.py)

- `_extract_ground_truth(resolution, question_type)` replaces `bool(answer)`
  - Binary: `bool(resolution.get("answer"))`
  - Continuous: `resolution.get("value")`
- Questions now carry `question_type` field through the pipeline

### Prompt Format

**Binary** (unchanged):
```
<<<PROBABILITIES>>>
0.65
0.42
<<<END>>>
```

**Continuous** (new):
```
<<<PERCENTILES>>>
Q1: p10=X, p25=Y, p50=Z, p75=A, p90=B
Q2: p10=X, p25=Y, p50=Z, p75=A, p90=B
<<<END>>>
```

### Parsing

- `parse_batch_percentiles(response, num_questions)` → `list[dict | None]`
- Each dict: `{"p10": float, "p25": float, "p50": float, "p75": float, "p90": float}`
- Falls back through: delimited block → Q-numbered lines → JSON → bare lines

### Metrics

- `compute_crps(percentiles, true_value)` — Quantile-weighted pinball loss approximation
- `compute_mae(p50, true_value)` — `|p50 - true_value|`

### Results Structure

```json
{
  "model_id": {
    "binary": {"brier_score": 0.21, "ece": 0.05, "n": 100, "by_template": {...}},
    "continuous": {"crps": 45.2, "mae": 38.7, "n": 50, "by_template": {...}}
  }
}
```

## Files Changed

| File | Change |
|------|--------|
| `src/civrealm/evaluation/sampling.py` | Fix ground_truth extraction, add question_type |
| `src/civrealm/evaluation/parallel_evaluator.py` | Continuous prompt, percentile parser, batch type dispatch |
| `src/civrealm/metrics.py` | Add compute_crps(), compute_mae() |
| `scripts/evaluate_llm_forecasts_parallel.py` | Split batches by type, dispatch metrics, update summary |

## What Doesn't Change

- models.py, rate_limiter.py, difficulty.py
- Checkpoint/resume, stratified sampling logic
- Question templates (already defined)
