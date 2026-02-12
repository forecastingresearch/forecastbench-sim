# LLM Evaluation Setup

This document describes how to configure and run LLM evaluations for CivBench forecasting experiments.

## API Key Configuration

CivBench uses [LiteLLM](https://github.com/BerriAI/litellm) for unified access to multiple LLM providers. API keys can be configured in two ways:

### Option 1: GCP Secret Manager (Recommended for Team Use)

API keys are stored in Google Cloud Platform Secret Manager and loaded automatically.

**Prerequisites:**
1. Access to the `civbench` GCP project
2. Google Cloud SDK installed (`gcloud`)

**Setup:**

```bash
# 1. Authenticate with GCP
gcloud auth application-default login

# 2. Verify your credentials work
gcloud auth application-default print-access-token

# 3. The .env file should have:
#    GOOGLE_CLOUD_PROJECT=civbench
#    GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

**Available Secrets:**
| Environment Variable | GCP Secret Name |
|---------------------|-----------------|
| `ANTHROPIC_API_KEY` | `API_KEY_ANTHROPIC` |
| `OPENAI_API_KEY` | `API_KEY_OPENAI` |
| `GOOGLE_API_KEY` | `API_KEY_GEMINI` |
| `TOGETHER_API_KEY` | `API_KEY_TOGETHERAI` |
| `MISTRAL_API_KEY` | `API_KEY_MISTRAL` |
| `XAI_API_KEY` | `API_KEY_XAI` |

**Reauthentication:**
If you see `Reauthentication is needed` errors:
```bash
gcloud auth application-default login
```

### Option 2: Environment Variables (Local Development)

Set API keys directly in your `.env` file:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-api...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
```

Or export them in your shell:
```bash
export ANTHROPIC_API_KEY=sk-ant-api...
```

## Running Evaluations

### Basic Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Run evaluation with specific models
python scripts/evaluate_llm_forecasts_parallel.py \
    --data-dir data/questions \
    --models "anthropic/claude-opus-4-5-20251101" \
    --questions-per-template 10 \
    --seed 42 \
    --output data/evaluations/results/my_eval.json
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data-dir` | Directory containing question banks | `data/questions` |
| `--models` | Model IDs in LiteLLM format | All frontier models |
| `-n`, `--questions-per-template` | Questions per template type | 10 |
| `--seed` | Random seed for reproducibility | 42 |
| `--output` | Output file path | Auto-generated |
| `--dry-run` | Load questions without querying | False |
| `--horizon` | Filter by horizon (H0-H7) | All |
| `--timeout` | Query timeout in seconds | 180 |

### Model ID Format

Models use LiteLLM format: `provider/model-name`

Examples:
- `anthropic/claude-opus-4-5-20251101`
- `anthropic/claude-sonnet-4-5-20250929`
- `openai/gpt-4o`
- `google/gemini-2.5-pro`

### Example: Conditional Framing Experiment

```bash
# Generate null conditional questions
python scripts/generate_null_conditional_questions.py \
    --input-dir data/questions \
    --output-dir data/conditional/null_conditional

# Evaluate on reframed questions
python scripts/evaluate_llm_forecasts_parallel.py \
    --data-dir data/conditional/null_conditional \
    --models "anthropic/claude-opus-4-5-20251101" \
    -n 10 --seed 42 \
    --output data/evaluations/results/null_conditional_eval.json
```

## Troubleshooting

### "Missing API Key" Error
1. Check GCP authentication: `gcloud auth application-default print-access-token`
2. If expired, reauthenticate: `gcloud auth application-default login`
3. Verify secrets exist: `gcloud secrets list --project=civbench`

### "No world report found" Warnings
Questions require a corresponding `world_report/` directory with game context. Games without world reports are skipped during evaluation.

### Quota Errors
The evaluation includes built-in rate limiting per provider. If you hit quota limits:
1. Reduce `--concurrent-batches` (default: 5)
2. Increase `--timeout` for slower models
3. Use `--checkpoint-interval` to save progress frequently

## Output Format

Evaluation results are saved as JSON:

```json
{
  "metadata": {
    "run_id": "20260203_114135",
    "seed": 42,
    "total_questions": 200,
    "base_rate": 0.435,
    "models": ["anthropic/claude-opus-4-5-20251101"]
  },
  "model_results": {
    "anthropic/claude-opus-4-5-20251101": {
      "brier_score": 0.1779,
      "ece": 0.0823,
      "num_predictions": 200,
      "brier_by_template": {...}
    }
  },
  "questions": [...]
}
```

## See Also

- [Conditional Forecasting](conditional_forecasting.md) - Conditional question experiments
- [Null Conditional Framing Experiment](../experiments/null_conditional_framing.md) - Framing effect study
