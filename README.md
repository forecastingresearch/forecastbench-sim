# CivBench

Forecasting benchmark for LLMs built on FreeCiv game simulations. Models read a world state report from a game in progress and answer questions about future game outcomes.

## What it tests

- **Binary questions** (20 templates): Will event X happen? Models output probability estimates, scored with Brier score and ECE.
- **Continuous questions** (6 templates): What will the value of Y be? Models output percentile estimates (p10, p25, p50, p75, p90), scored with CRPS and MAE.
- **Conditional reasoning**: Given a hypothetical intervention (government change, treasury boost), can the model update its forecasts? Five framing variants test different aspects of causal reasoning.

The benchmark contains 9,426 questions across 26 templates, 14 game seeds, and 8 forecast horizons (0--210 turns).

## Quick start

```bash
# Install
uv sync

# Dry run (no API calls)
uv run python scripts/evaluate_llm_forecasts_parallel.py --dry-run -n 2

# Evaluate a model (2 questions per template)
uv run python scripts/evaluate_llm_forecasts_parallel.py \
  --models anthropic/claude-sonnet-4-5-20250929 -n 2

# Continuous questions only
uv run python scripts/evaluate_llm_forecasts_parallel.py \
  --models anthropic/claude-sonnet-4-5-20250929 -n 2 \
  --question-type continuous

# Filter by forecast horizon
uv run python scripts/evaluate_llm_forecasts_parallel.py \
  --models openai/gpt-4o -n 5 --horizon H0 H1
```

Results are saved to `data/evaluations/runs/`.

## Pipeline

```
Game Execution → Serialization → Question Generation → Conditional Experiments → LLM Evaluation
(FreeCiv)          (games/)        (questions/)          (conditional/)           (evaluations/)
```

| Stage | Script | Purpose |
|-------|--------|---------|
| Game execution | `scripts/run_world.py` | Run a single FreeCiv simulation |
| Game execution | `scripts/run_worlds.py` | Run batch of simulations |
| Game execution | `scripts/run_fork.py` | Fork a game with an intervention |
| Serialization | `scripts/generate_data_batch.py` | Serialize game saves to JSON |
| Questions | `scripts/generate_questions.py` | Generate questions for one game |
| Questions | `scripts/generate_questions_batch.py` | Generate questions across seeds |
| Conditional | `scripts/setup_republic_conditional_eval.py` | Set up republic conditional framings |
| Conditional | `scripts/setup_gold500_conditional_eval.py` | Set up gold500 conditional framings |
| Evaluation | `scripts/evaluate_llm_forecasts_parallel.py` | Run LLM evaluations in parallel |
| Analysis | `scripts/analyze_conditional_comparison.py` | Compare conditional vs. baseline |
| Analysis | `scripts/compute_difficulty_scores.py` | Compute question difficulty |
| Analysis | `scripts/plot_conditional_framing_comparison.py` | Plot framing comparison results |

## Data layout

```
data/
├── games/                  # Serialized game state JSON (base seeds + forks)
├── questions/              # Generated forecasting questions + world reports
├── conditional/            # Conditional forecasting experiments
│   ├── republic/           # Government-change intervention (5 framings)
│   ├── gold500/            # Treasury-boost intervention (2 framings)
│   └── null_conditional/   # Framing placebo control
└── evaluations/
    ├── runs/               # Raw per-run evaluation JSON
    ├── results/            # Aggregated evaluation results
    └── plots/              # Visualizations (PNG, PDF)
```

See `data/README.md` and `data/conditional/README.md` for details.

## Evaluation output

Each run produces a JSON file with:

- **`model_results`**: Per-model metrics split by question type
  - `binary`: `brier_score`, `ece`, per-template Brier breakdown
  - `continuous`: `crps`, `mae`, per-template CRPS/MAE breakdown
- **`questions`**: Per-question predictions with `question_type`, ground truth, and model outputs
- **`metadata`**: Run config, question type distribution, template distribution

## Citation

This project extends [CivRealm](https://github.com/bigai-ai/civrealm) (Qi et al., ICLR 2024) as a forecasting benchmark.

```bibtex
@inproceedings{qi2024civrealm,
  title     = {CivRealm: A Learning and Reasoning Odyssey in Civilization for Decision-Making Agents},
  author    = {Siyuan Qi and Shuo Chen and Yexin Li and Xiangyu Kong and Junqi Wang and Bangcheng Yang and Pring Wong and Yifan Zhong and Xiaoyuan Zhang and Zhaowei Zhang and Nian Liu and Wei Wang and Yaodong Yang and Song-Chun Zhu},
  booktitle = {International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=UBVNwD3hPN}
}
```
