#!/bin/bash
# Stage 3: Run forks for seeds 11-20 with 3 interventions each, in parallel.
# Resumes from where generate_test_seeds.sh left off after Stage 2.
#
# Usage:
#   cd ~/Projects/civbench
#   bash scripts/run_test_forks.sh 2>&1 | tee run_test_forks.log

set -e

cd "$(dirname "$0")/.."  # cd to project root

START_SEED=11
END_SEED=20
CHECKPOINT=60
END_TURN=300
PARALLEL=5

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

run_single_fork() {
    local SEED=$1
    local MOD=$2
    local FORK_NAME=$3
    local CONDITION=$4
    local FORK_DIR="logs/recordings/seed${SEED}fork${FORK_NAME}"

    if [ -f "$FORK_DIR/conditional_results.json" ]; then
        echo "[seed${SEED}/${FORK_NAME}] Already exists — skipping"
        return 0
    fi

    echo "[seed${SEED}/${FORK_NAME}] Running fork..."
    if ! uv run python scripts/run_fork.py \
        --base-seed $SEED \
        --checkpoint-turn $CHECKPOINT \
        --end-turn $END_TURN \
        --modification "$MOD" \
        --fork-name "$FORK_NAME" \
        --recording-dir "logs/recordings/seed${SEED}" 2>&1 | tail -5; then
        echo "[seed${SEED}/${FORK_NAME}] Fork FAILED"
        return 1
    fi

    echo "[seed${SEED}/${FORK_NAME}] Generating conditional results..."
    uv run python scripts/generate_conditional_results.py \
        --baseline-dir "logs/recordings/seed${SEED}" \
        --fork-dir "$FORK_DIR" \
        --condition "$CONDITION" \
        --checkpoint-turn $CHECKPOINT \
        --end-turn $END_TURN \
        --game-data "data/games/seed${SEED}_data.json" 2>&1 | tail -5

    # Clean up large state/action JSONs to save disk
    find "$FORK_DIR" -maxdepth 1 -name 'turn_*_state.json' -delete 2>/dev/null
    find "$FORK_DIR" -maxdepth 1 -name 'turn_*_available_action.json' -delete 2>/dev/null
    # Remove savegames (conditional_results.json has everything we need)
    rm -rf "$FORK_DIR/savegames" 2>/dev/null

    echo "[seed${SEED}/${FORK_NAME}] Done"
}

run_intervention_batch() {
    local INTERVENTION_NAME=$1
    local MOD=$2
    local FORK_NAME=$3
    local CONDITION=$4

    log "=== ${INTERVENTION_NAME} forks ==="
    for ((i=$START_SEED; i<=$END_SEED; i+=PARALLEL)); do
        PIDS=()
        SEEDS=()
        for ((j=i; j<i+PARALLEL && j<=END_SEED; j++)); do
            run_single_fork $j "$MOD" "$FORK_NAME" "$CONDITION" &
            PIDS+=($!)
            SEEDS+=($j)
        done
        log "Batch: seeds ${SEEDS[*]}"
        for PID in "${PIDS[@]}"; do wait $PID || true; done
        log "Batch done"
    done
    log "=== ${INTERVENTION_NAME} COMPLETE ==="
    echo
}

run_intervention_batch "Republic" "government:0:Republic" "govRepublicp0" "government:0:Republic"
run_intervention_batch "Gold +500" "gold_add:0:500" "goldadd500p0" "gold_add:0:500"
run_intervention_batch "Map Making" "tech:0:45" "tech45p0" "tech:0:45"

# Summary
log "=== SUMMARY ==="
for SEED in $(seq $START_SEED $END_SEED); do
    R=$([ -f "logs/recordings/seed${SEED}forkgovRepublicp0/conditional_results.json" ] && echo "✓" || echo "✗")
    G=$([ -f "logs/recordings/seed${SEED}forkgoldadd500p0/conditional_results.json" ] && echo "✓" || echo "✗")
    M=$([ -f "logs/recordings/seed${SEED}forktech45p0/conditional_results.json" ] && echo "✓" || echo "✗")
    printf "  seed%-2d: Republic=%s  Gold=%s  MapMaking=%s\n" $SEED "$R" "$G" "$M"
done
