#!/bin/bash
cd /Users/elsehow/Projects/civbench
source .venv/bin/activate
python scripts/evaluate_llm_forecasts_parallel.py \
    --data-dir data/conditional/null_conditional \
    --models "anthropic/claude-opus-4-5-20251101" \
    --questions-per-template 10 \
    --seed 42 \
    --output data/evaluations/null_conditional_opus45_eval.json
