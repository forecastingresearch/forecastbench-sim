#!/usr/bin/env bash
set -euo pipefail
MAX_JOBS=${MAX_JOBS:-3}
LAUNCH_DELAY_SECONDS=${LAUNCH_DELAY_SECONDS:-2}
COMMON="--all --checkpoint-interval 1 --timeout 900"

cleanup() {
  local pids
  pids="$(jobs -pr || true)"
  if [ -n "$pids" ]; then
    echo "Stopping background eval jobs..."
    kill $pids 2>/dev/null || true
    wait $pids 2>/dev/null || true
  fi
}

trap 'cleanup; exit 130' INT TERM
trap 'cleanup' EXIT

runs=(
"cont_uncond|--question-type continuous -o data/evaluations/runs/cont_uncond_all.json"
"rep_base_bin|--data-dir data/conditional/republic/baseline --question-type binary -o data/evaluations/runs/republic_baseline_binary_all.json"
"rep_base_cont|--data-dir data/conditional/republic/baseline --question-type continuous -o data/evaluations/runs/republic_baseline_continuous_all.json"
"rep_cond_bin|--data-dir data/conditional/republic/conditional --question-type binary -o data/evaluations/runs/republic_conditional_binary_all.json"
"rep_cond_cont|--data-dir data/conditional/republic/conditional --question-type continuous -o data/evaluations/runs/republic_conditional_continuous_all.json"
"gold_base_bin|--data-dir data/conditional/gold500/baseline --question-type binary -o data/evaluations/runs/gold500_baseline_binary_all.json"
"gold_base_cont|--data-dir data/conditional/gold500/baseline --question-type continuous -o data/evaluations/runs/gold500_baseline_continuous_all.json"
"gold_cond_bin|--data-dir data/conditional/gold500/conditional --question-type binary -o data/evaluations/runs/gold500_conditional_binary_all.json"
"gold_cond_cont|--data-dir data/conditional/gold500/conditional --question-type continuous -o data/evaluations/runs/gold500_conditional_continuous_all.json"
)

for run in "${runs[@]}"; do
  name="${run%%|*}"
  args="${run#*|}"
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 3; done
  (
    uv run python scripts/evaluate_llm_forecasts_parallel.py $args $COMMON
  ) > "logs/runner_${name}.log" 2>&1 &
  sleep "$LAUNCH_DELAY_SECONDS"
done
wait
