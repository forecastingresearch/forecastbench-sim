#!/usr/bin/env bash
set -euo pipefail

# Retry only Anthropic failures from existing checkpointed runs.
#
# Default behavior:
# - Detect runs where Anthropic predictions still contain workspace limit errors
# - Resume from that run's checkpoint
# - Re-run only Anthropic models
# - Write to staged JSONs (does not overwrite canonical outputs)
#
# Optional behavior:
# - Set PROMOTE_ON_SUCCESS=1 to overwrite canonical output when no limit errors remain.
# - Set WRITE_MODE=inplace to write directly to canonical output path.
#
# Example:
#   bash retry-anthropic-failed.sh
#   PROMOTE_ON_SUCCESS=1 bash retry-anthropic-failed.sh
#   MAX_JOBS=2 BATCH_CONCURRENCY=5 bash retry-anthropic-failed.sh

MAX_JOBS=${MAX_JOBS:-3}
LAUNCH_DELAY_SECONDS=${LAUNCH_DELAY_SECONDS:-2}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-1}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-900}
BATCH_CONCURRENCY=${BATCH_CONCURRENCY:-5}
WRITE_MODE=${WRITE_MODE:-staged}           # staged | inplace
PROMOTE_ON_SUCCESS=${PROMOTE_ON_SUCCESS:-0} # 0 | 1 (only used in staged mode)
DRY_RUN=${DRY_RUN:-0}                       # 0 | 1
LIMIT_MSG=${LIMIT_MSG:-workspace API usage limits}

ANTHROPIC_MODEL_1="anthropic/claude-opus-4-5-20251101"
ANTHROPIC_MODEL_2="anthropic/claude-sonnet-4-5-20250929"

# name|data_dir|question_type|canonical_output_json
runs=(
"cont_uncond|data/questions|continuous|data/evaluations/runs/cont_uncond_all.json"
"rep_base_bin|data/conditional/republic/baseline|binary|data/evaluations/runs/republic_baseline_binary_all.json"
"rep_base_cont|data/conditional/republic/baseline|continuous|data/evaluations/runs/republic_baseline_continuous_all.json"
"rep_cond_bin|data/conditional/republic/conditional|binary|data/evaluations/runs/republic_conditional_binary_all.json"
"rep_cond_cont|data/conditional/republic/conditional|continuous|data/evaluations/runs/republic_conditional_continuous_all.json"
"gold_base_bin|data/conditional/gold500/baseline|binary|data/evaluations/runs/gold500_baseline_binary_all.json"
"gold_base_cont|data/conditional/gold500/baseline|continuous|data/evaluations/runs/gold500_baseline_continuous_all.json"
"gold_cond_bin|data/conditional/gold500/conditional|binary|data/evaluations/runs/gold500_conditional_binary_all.json"
"gold_cond_cont|data/conditional/gold500/conditional|continuous|data/evaluations/runs/gold500_conditional_continuous_all.json"
)

cleanup() {
  local pids
  pids="$(jobs -pr || true)"
  if [ -n "$pids" ]; then
    echo "Stopping background retry jobs..."
    kill $pids 2>/dev/null || true
    wait $pids 2>/dev/null || true
  fi
}
trap 'cleanup; exit 130' INT TERM
trap 'cleanup' EXIT

count_limit_errors() {
  local json_path="$1"
  python3 - "$json_path" "$LIMIT_MSG" "$ANTHROPIC_MODEL_1" "$ANTHROPIC_MODEL_2" <<'PY'
import json
import sys

path, needle, m1, m2 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(path) as f:
    data = json.load(f)

models = [m1, m2]
count = 0
for q in data.get("questions", []):
    preds = q.get("predictions", {})
    for m in models:
        err = str(preds.get(m, {}).get("error") or "")
        if needle in err:
            count += 1
print(count)
PY
}

get_run_id() {
  local json_path="$1"
  python3 - "$json_path" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
run_id = data.get("metadata", {}).get("run_id")
if not run_id:
    raise SystemExit(1)
print(run_id)
PY
}

staged_output_path() {
  local canonical="$1"
  local stem="${canonical%.json}"
  echo "${stem}.anthropic_retry.json"
}

launch_retry() {
  local name="$1"
  local data_dir="$2"
  local qtype="$3"
  local canonical_out="$4"
  local staged_out="$5"
  local resume_base="$6"
  local log_file="$7"

  local out_path="$canonical_out"
  if [ "$WRITE_MODE" = "staged" ]; then
    out_path="$staged_out"
  fi

  local cmd=(
    uv run python scripts/evaluate_llm_forecasts_parallel.py
    --data-dir "$data_dir"
    --question-type "$qtype"
    --all
    --checkpoint-interval "$CHECKPOINT_INTERVAL"
    --timeout "$TIMEOUT_SECONDS"
    --concurrent-batches "$BATCH_CONCURRENCY"
    --models "$ANTHROPIC_MODEL_1" "$ANTHROPIC_MODEL_2"
    --resume "$resume_base"
    -o "$out_path"
  )

  if [ "$DRY_RUN" = "1" ]; then
    {
      printf '[DRY_RUN] %s\n' "$name"
      printf '  %q ' "${cmd[@]}"
      printf '\n'
    } | tee "$log_file"
    return 0
  fi

  (
    "${cmd[@]}"

    local after
    after="$(count_limit_errors "$out_path")"
    echo "[$name] remaining anthropic limit errors in output: $after"

    if [ "$WRITE_MODE" = "staged" ] && [ "$PROMOTE_ON_SUCCESS" = "1" ] && [ "$after" = "0" ]; then
      cp "$out_path" "$canonical_out"
      echo "[$name] promoted staged output -> $canonical_out"
    fi
  ) >"$log_file" 2>&1
}

echo "Starting Anthropic failed-batch retries"
echo "  MAX_JOBS=$MAX_JOBS WRITE_MODE=$WRITE_MODE PROMOTE_ON_SUCCESS=$PROMOTE_ON_SUCCESS"
echo "  BATCH_CONCURRENCY=$BATCH_CONCURRENCY TIMEOUT_SECONDS=$TIMEOUT_SECONDS"

for run in "${runs[@]}"; do
  name="${run%%|*}"
  rest="${run#*|}"
  data_dir="${rest%%|*}"
  rest="${rest#*|}"
  qtype="${rest%%|*}"
  canonical_out="${rest#*|}"

  if [ ! -f "$canonical_out" ]; then
    echo "[$name] skip: missing output file $canonical_out"
    continue
  fi

  before="$(count_limit_errors "$canonical_out")"
  if [ "$before" = "0" ]; then
    echo "[$name] skip: no Anthropic workspace-limit errors"
    continue
  fi

  run_id="$(get_run_id "$canonical_out")"
  resume_dir="logs/eval_${run_id}"
  resume_base="${resume_dir}/checkpoint.json"
  typed_checkpoint="${resume_dir}/checkpoint_${qtype}.json"

  if [ ! -f "$typed_checkpoint" ]; then
    echo "[$name] skip: missing checkpoint file $typed_checkpoint"
    continue
  fi

  staged_out="$(staged_output_path "$canonical_out")"
  log_file="logs/retry_runner_${name}.log"

  echo "[$name] retrying (before_errors=$before)"
  echo "  data_dir=$data_dir qtype=$qtype"
  echo "  resume=$resume_base"
  if [ "$WRITE_MODE" = "staged" ]; then
    echo "  output(staged)=$staged_out"
  else
    echo "  output(in-place)=$canonical_out"
  fi
  echo "  log=$log_file"

  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 2; done
  launch_retry "$name" "$data_dir" "$qtype" "$canonical_out" "$staged_out" "$resume_base" "$log_file" &
  sleep "$LAUNCH_DELAY_SECONDS"
done

wait

echo "Retry runner complete."
if [ "$WRITE_MODE" = "staged" ] && [ "$PROMOTE_ON_SUCCESS" != "1" ]; then
  echo "Staged outputs were not promoted. Set PROMOTE_ON_SUCCESS=1 to auto-promote."
fi
