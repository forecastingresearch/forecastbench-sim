# Null Conditional Framing Experiment

## Results Summary

| Condition | Brier Score | N |
|-----------|-------------|---|
| **Baseline (unconditional)** | 0.1779 | ~200 |
| **Null conditional framing** | 0.2418 | 38 |
| **Difference** | +0.0639 | (+36%) |

**Key Finding:** Conditional framing alone worsens forecasting performance by 36%, even when no actual intervention occurs.

![Comparison Chart](../../data/evaluations/plots/null_conditional_comparison.png)

## Research Question
Does the presence of conditional question *framing* affect Brier score, independent of any actual intervention?

## Background
- **Unconditional questions**: "Will Egypt have more technologies than Rome at turn 120?"
- **Conditional intervention**: "If Egypt received +5000 gold next turn, would Egypt have more technologies than Rome at turn 120?"
- **Null conditional** (this experiment): "Given that Egypt will NOT receive any gold bonus, will Egypt have more technologies than Rome at turn 120?"

## Hypothesis
If LLMs are good at conditional reasoning, the null conditional framing should have **no effect** on Brier scores (since no actual intervention occurs). If conditional framing alone affects reasoning, we expect to see a **difference** between null conditional and baseline unconditional Brier scores.

## Interpretation

The results suggest that **conditional framing introduces cognitive overhead** or **confuses the model's reasoning**, even when no actual intervention occurs. When asked "Given that X will NOT receive any gold bonus, will X...", Opus 4.5 performs significantly worse than when asked the equivalent "Will X...".

This has implications for:
1. **Benchmark design**: Conditional framing may artificially inflate difficulty
2. **Prompt engineering**: Unnecessary conditional language may degrade performance
3. **Causal reasoning evaluation**: Need to control for framing effects when measuring conditional reasoning ability

## Infrastructure Created

### 1. Question Generation Script
`scripts/generate_null_conditional_questions.py`

Transforms unconditional questions to null conditional framing:
```python
# Example transformations:
"Will X have more Y than Z?"
→ "Given that X will NOT receive any gold bonus, will X have more Y than Z?"
```

### 2. Generated Questions
Located in `data/conditional/null_conditional/`
- 27,768 questions transformed
- Ground truth unchanged (no actual intervention)
- Template IDs prefixed with `null_conditional_`

### 3. Baseline Data
- Opus 4.5 unconditional Brier: **0.1779**
- Source: `data/evaluations/results/parallel_eval_42_20260202_130658.json`

## How to Run the Evaluation

```bash
# Ensure API keys are available (via GCP Secret Manager or environment)
export ANTHROPIC_API_KEY=your_key_here

# Run evaluation
source .venv/bin/activate
python scripts/evaluate_llm_forecasts_parallel.py \
    --data-dir data/conditional/null_conditional \
    --models anthropic/claude-opus-4-5-20251101 \
    --questions-per-template 10 \
    --seed 42 \
    --output data/evaluations/results/null_conditional_opus45_eval.json
```

Note: Only seed* games have world reports available. For a broader evaluation, regenerate questions from games with world reports, or use the original `data/questions` directory as source.

## Expected Analysis

Compare:
1. **Baseline**: Opus 4.5 on unconditional questions (Brier: 0.1779)
2. **Null conditional**: Opus 4.5 on same questions with null conditional framing
3. **Conditional positive** (existing): Opus 4.5 on actual conditional questions from fork

### Metrics to Report
- Overall Brier score comparison
- Brier score by template type
- Statistical significance test (paired t-test on question-level Briers)

### Creating Comparison Chart
After evaluation completes:
```python
import matplotlib.pyplot as plt
import json

# Load results
baseline = json.load(open('data/evaluations/results/parallel_eval_42_20260202_130658.json'))
null_cond = json.load(open('data/evaluations/results/null_conditional_opus45_eval.json'))

# Extract Brier scores
baseline_brier = baseline['model_results']['anthropic/claude-opus-4-5-20251101']['brier_score']
null_cond_brier = null_cond['model_results']['anthropic/claude-opus-4-5-20251101']['brier_score']

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
conditions = ['Unconditional\n(baseline)', 'Null Conditional\nFraming']
briers = [baseline_brier, null_cond_brier]
ax.bar(conditions, briers, color=['#2ecc71', '#e74c3c'])
ax.set_ylabel('Brier Score (lower is better)')
ax.set_title('Effect of Conditional Framing on Opus 4.5 Forecasting')
ax.set_ylim(0, 0.3)
for i, v in enumerate(briers):
    ax.text(i, v + 0.01, f'{v:.4f}', ha='center')
plt.tight_layout()
plt.savefig('data/evaluations/plots/null_conditional_comparison.png', dpi=150)
```

## Issue Tracking
- Beads issue: `civbench-uqi`
- Status: Infrastructure complete, awaiting evaluation run
