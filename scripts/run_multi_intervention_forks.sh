#!/bin/bash
# Run multi-intervention forks for the N-Intervention Scaling experiment.
#
# Generates compound-intervention forks on all 21 seeds (0-20):
#   - N=1 forks for seeds 0-10 (Republic, Gold+500, Map Making) — these don't exist yet
#   - N=2 forks for seeds 0-20 (Republic + Gold+500)
#   - N=3 forks for seeds 0-20 (Republic + Gold+500 + Map Making)
#
# Each fork: checkpoint T60, end T150, resolution turns 90/120/150 (H1/H2/H3).
#
# Prerequisites:
#   - Baseline games exist for all seeds (logs/recordings/seed{N}/)
#   - Game data JSONs exist (data/games/seed{N}_data.json)
#   - freeciv-web Docker container running
#
# Usage:
#   cd ~/Projects/civbench
#   bash scripts/run_multi_intervention_forks.sh 2>&1 | tee run_multi_intervention_forks.log

set -e

cd "$(dirname "$0")/.."  # cd to project root

CHECKPOINT=60
END_TURN=150
PARALLEL=5  # concurrent forks

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

run_fork_and_generate() {
    local SEED=$1
    local MODS=$2       # space-separated modification args for run_fork.py
    local FORK_NAME=$3
    local CONDITION=$4   # condition arg for generate_conditional_results.py
    local FORK_DIR="logs/recordings/seed${SEED}fork${FORK_NAME}"

    if [ -d "$FORK_DIR" ] && [ -f "$FORK_DIR/conditional_results.json" ]; then
        log "[seed${SEED}/${FORK_NAME}] Already exists — skipping"
        return 0
    fi

    log "[seed${SEED}/${FORK_NAME}] Running fork..."

    # Build modification args array
    local MOD_ARGS=""
    for MOD in $MODS; do
        MOD_ARGS="$MOD_ARGS --modification $MOD"
    done

    if ! uv run python scripts/run_fork.py \
        --base-seed "$SEED" \
        --checkpoint-turn $CHECKPOINT \
        --end-turn $END_TURN \
        $MOD_ARGS \
        --fork-name "$FORK_NAME" \
        --recording-dir "logs/recordings/seed${SEED}" 2>&1; then
        log "[seed${SEED}/${FORK_NAME}] Fork FAILED"
        return 1
    fi

    log "[seed${SEED}/${FORK_NAME}] Generating conditional results..."
    uv run python scripts/generate_conditional_results.py \
        --baseline-dir "logs/recordings/seed${SEED}" \
        --fork-dir "$FORK_DIR" \
        --condition "$CONDITION" \
        --checkpoint-turn $CHECKPOINT \
        --end-turn $END_TURN \
        --game-data "data/games/seed${SEED}_data.json" \
        --verbose

    log "[seed${SEED}/${FORK_NAME}] Done"
}

run_batch() {
    # Run a batch of seeds for a given fork config
    local START=$1
    local END=$2
    local MODS=$3
    local FORK_NAME=$4
    local CONDITION=$5

    for ((i=START; i<=END; i+=PARALLEL)); do
        BATCH_PIDS=()
        BATCH_SEEDS=()
        for ((j=i; j<=END && j<i+PARALLEL; j++)); do
            BATCH_SEEDS+=($j)
            run_fork_and_generate $j "$MODS" "$FORK_NAME" "$CONDITION" &
            BATCH_PIDS+=($!)
        done
        log "Waiting for batch: seeds ${BATCH_SEEDS[*]}"
        local FAIL=0
        for PID in "${BATCH_PIDS[@]}"; do
            if ! wait $PID; then FAIL=1; fi
        done
        if [ $FAIL -eq 1 ]; then
            log "WARNING: Some forks in batch failed. Continuing..."
        fi
    done
}

# ============================================================
# Stage 1: N=1 forks for seeds 0-10 (fill the gap)
# ============================================================
log "=== STAGE 1: N=1 forks for seeds 0-10 ==="
log "These are needed for interaction analysis (superadditivity test)."
echo

log "--- Stage 1a: Republic forks (seeds 0-10) ---"
run_batch 0 10 "government:0:Republic" "govRepublicp0" "government:0:Republic"
log "--- Stage 1a COMPLETE ---"
echo

log "--- Stage 1b: Gold +500 forks (seeds 0-10) ---"
run_batch 0 10 "gold_add:0:500" "goldadd500p0" "gold_add:0:500"
log "--- Stage 1b COMPLETE ---"
echo

log "--- Stage 1c: Map Making forks (seeds 0-10) ---"
run_batch 0 10 "tech:0:45" "tech45p0" "tech:0:45"
log "--- Stage 1c COMPLETE ---"
echo

# ============================================================
# Stage 2: N=2 forks (Republic + Gold) for seeds 0-20
# ============================================================
log "=== STAGE 2: N=2 forks (Republic + Gold+500) for seeds 0-20 ==="
run_batch 0 20 "government:0:Republic gold_add:0:500" "govRepublic_goldadd500_p0" "government:0:Republic"
log "=== STAGE 2 COMPLETE ==="
echo

# ============================================================
# Stage 3: N=3 forks (Republic + Gold + Map Making) for seeds 0-20
# ============================================================
log "=== STAGE 3: N=3 forks (Republic + Gold+500 + Map Making) for seeds 0-20 ==="
run_batch 0 20 "government:0:Republic gold_add:0:500 tech:0:45" "govRepublic_goldadd500_tech45_p0" "government:0:Republic"
log "=== STAGE 3 COMPLETE ==="
echo

# ============================================================
# Summary
# ============================================================
log "=== ALL STAGES COMPLETE ==="
echo

log "N=1 forks (seeds 0-10):"
for SEED in $(seq 0 10); do
    REPUBLIC="logs/recordings/seed${SEED}forkgovRepublicp0/conditional_results.json"
    GOLD="logs/recordings/seed${SEED}forkgoldadd500p0/conditional_results.json"
    MAPMAKING="logs/recordings/seed${SEED}forktech45p0/conditional_results.json"
    printf "  seed%-2d: Republic=%s  Gold=%s  MapMaking=%s\n" \
        $SEED \
        $([ -f "$REPUBLIC" ] && echo "✓" || echo "✗") \
        $([ -f "$GOLD" ] && echo "✓" || echo "✗") \
        $([ -f "$MAPMAKING" ] && echo "✓" || echo "✗")
done

echo
log "N=2 forks (Republic + Gold, seeds 0-20):"
for SEED in $(seq 0 20); do
    CR="logs/recordings/seed${SEED}forkgovRepublic_goldadd500_p0/conditional_results.json"
    printf "  seed%-2d: %s\n" $SEED $([ -f "$CR" ] && echo "✓" || echo "✗")
done

echo
log "N=3 forks (Republic + Gold + MapMaking, seeds 0-20):"
for SEED in $(seq 0 20); do
    CR="logs/recordings/seed${SEED}forkgovRepublic_goldadd500_tech45_p0/conditional_results.json"
    printf "  seed%-2d: %s\n" $SEED $([ -f "$CR" ] && echo "✓" || echo "✗")
done

echo
log "Next steps:"
log "  1. Run: uv run python scripts/setup_multi_intervention_eval.py"
log "  2. Spot-check interaction effects in ground truth"
log "  3. Run LLM evaluations on N=2 and N=3 conditions"
