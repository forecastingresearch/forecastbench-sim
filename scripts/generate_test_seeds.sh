#!/bin/bash
# Generate 10 new test seeds (11-20) with 3 interventions each
# for the calibration surgery experiment.
#
# Runs in parallel where possible:
#   Stage 1: 10 baseline games in parallel (batch_size=5)
#   Stage 2: Extract game data (already parallel internally)
#   Stage 3: Fork each seed × 3 interventions in parallel
#   Stage 4: Summary + next steps
#
# Usage:
#   cd ~/Projects/civbench
#   bash scripts/generate_test_seeds.sh 2>&1 | tee generate_test_seeds.log

set -e

cd "$(dirname "$0")/.."  # cd to project root

START_SEED=11
END_SEED=20
CHECKPOINT=60
END_TURN=300
MAX_TURNS=300
PARALLEL=5  # concurrent games

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

# ============================================================
# Stage 1: Run baseline games IN PARALLEL
# ============================================================
log "=== STAGE 1: Running baseline games (seeds $START_SEED-$END_SEED, ${PARALLEL} parallel, ${MAX_TURNS} turns) ==="

# Check which seeds already exist
SEEDS_TO_RUN=""
for SEED in $(seq $START_SEED $END_SEED); do
    RECORDING_DIR="logs/recordings/seed${SEED}"
    if [ -d "$RECORDING_DIR/savegames" ]; then
        COUNT=$(ls $RECORDING_DIR/savegames/*.sav.xz 2>/dev/null | wc -l)
        if [ "$COUNT" -gt 250 ]; then
            log "[seed${SEED}] Baseline already exists ($COUNT savegames) — skipping"
            continue
        fi
    fi
    SEEDS_TO_RUN="$SEEDS_TO_RUN $SEED"
done

if [ -n "$SEEDS_TO_RUN" ]; then
    SEEDS_ARRAY=($SEEDS_TO_RUN)
    TOTAL=${#SEEDS_ARRAY[@]}
    log "Running $TOTAL baseline games: ${SEEDS_TO_RUN}"

    # Run in batches of $PARALLEL
    for ((i=0; i<TOTAL; i+=PARALLEL)); do
        BATCH_PIDS=()
        BATCH_SEEDS=()
        for ((j=i; j<i+PARALLEL && j<TOTAL; j++)); do
            SEED=${SEEDS_ARRAY[$j]}
            BATCH_SEEDS+=($SEED)
            log "[seed${SEED}] Starting baseline game..."
            uv run python scripts/run_world.py --seed $SEED --max_turns $MAX_TURNS --quiet &
            BATCH_PIDS+=($!)
        done

        log "Waiting for batch: seeds ${BATCH_SEEDS[*]} (PIDs: ${BATCH_PIDS[*]})"
        FAIL=0
        for idx in "${!BATCH_PIDS[@]}"; do
            PID=${BATCH_PIDS[$idx]}
            SEED=${BATCH_SEEDS[$idx]}
            if wait $PID; then
                COUNT=$(ls logs/recordings/seed${SEED}/savegames/*.sav.xz 2>/dev/null | wc -l)
                log "[seed${SEED}] Done — $COUNT savegames"
            else
                log "[seed${SEED}] FAILED (exit code $?)"
                FAIL=1
            fi
        done
        if [ $FAIL -eq 1 ]; then
            log "WARNING: Some games in batch failed. Continuing..."
        fi
    done
else
    log "All baselines already exist."
fi

log "=== STAGE 1 COMPLETE ==="
echo

# ============================================================
# Stage 2: Extract game data JSON
# ============================================================
log "=== STAGE 2: Extracting game data ==="

mkdir -p data/games

uv run python scripts/generate_data_batch.py \
    --input-dir logs/recordings \
    --output-dir data/games \
    --workers 4

for SEED in $(seq $START_SEED $END_SEED); do
    if [ ! -f "data/games/seed${SEED}_data.json" ]; then
        log "WARNING: Missing game data for seed${SEED}!"
    fi
done

log "=== STAGE 2 COMPLETE ==="
echo

# ============================================================
# Stage 3: Fork each seed × 3 interventions IN PARALLEL
# ============================================================

run_fork_and_generate() {
    local SEED=$1
    local MOD=$2
    local FORK_NAME=$3
    local CONDITION=$4
    local FORK_DIR="logs/recordings/seed${SEED}fork${FORK_NAME}"

    if [ -d "$FORK_DIR" ] && [ -f "$FORK_DIR/conditional_results.json" ]; then
        log "[seed${SEED}/${FORK_NAME}] Already exists — skipping"
        return 0
    fi

    log "[seed${SEED}/${FORK_NAME}] Running fork..."
    if ! uv run python scripts/run_fork.py \
        --base-seed $SEED \
        --checkpoint-turn $CHECKPOINT \
        --end-turn $END_TURN \
        --modification "$MOD" \
        --fork-name "$FORK_NAME" \
        --recording-dir "logs/recordings/seed${SEED}" 2>&1; then
        log "[seed${SEED}/${FORK_NAME}] Fork FAILED — skipping"
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

# --- 3a: Republic forks (parallel by seed) ---
log "=== STAGE 3a: Republic forks ==="
for ((i=$START_SEED; i<=$END_SEED; i+=PARALLEL)); do
    BATCH_PIDS=()
    for ((j=i; j<i+PARALLEL && j<=END_SEED; j++)); do
        run_fork_and_generate $j "government:0:Republic" "govRepublicp0" "government:0:Republic" &
        BATCH_PIDS+=($!)
    done
    for PID in "${BATCH_PIDS[@]}"; do wait $PID; done
done
log "=== STAGE 3a COMPLETE ==="
echo

# --- 3b: Gold +500 forks (parallel by seed) ---
log "=== STAGE 3b: Gold +500 forks ==="
for ((i=$START_SEED; i<=$END_SEED; i+=PARALLEL)); do
    BATCH_PIDS=()
    for ((j=i; j<i+PARALLEL && j<=END_SEED; j++)); do
        run_fork_and_generate $j "gold_add:0:500" "goldadd500p0" "gold_add:0:500" &
        BATCH_PIDS+=($!)
    done
    for PID in "${BATCH_PIDS[@]}"; do wait $PID; done
done
log "=== STAGE 3b COMPLETE ==="
echo

# --- 3c: Map Making forks (parallel by seed) ---
log "=== STAGE 3c: Map Making forks ==="
for ((i=$START_SEED; i<=$END_SEED; i+=PARALLEL)); do
    BATCH_PIDS=()
    for ((j=i; j<i+PARALLEL && j<=END_SEED; j++)); do
        run_fork_and_generate $j "tech:0:45" "tech45p0" "tech:0:45" &
        BATCH_PIDS+=($!)
    done
    for PID in "${BATCH_PIDS[@]}"; do wait $PID; done
done
log "=== STAGE 3c COMPLETE ==="
echo

# ============================================================
# Summary
# ============================================================
log "=== ALL STAGES COMPLETE ==="
log ""
log "Baseline games:"
for SEED in $(seq $START_SEED $END_SEED); do
    DIR="logs/recordings/seed${SEED}"
    if [ -d "$DIR/savegames" ]; then
        echo "  seed${SEED}: $(ls $DIR/savegames/*.sav.xz 2>/dev/null | wc -l) savegames"
    else
        echo "  seed${SEED}: MISSING"
    fi
done

log ""
log "Forks (conditional_results.json present):"
for SEED in $(seq $START_SEED $END_SEED); do
    REPUBLIC="logs/recordings/seed${SEED}forkgovRepublicp0/conditional_results.json"
    GOLD="logs/recordings/seed${SEED}forkgoldadd500p0/conditional_results.json"
    MAPMAKING="logs/recordings/seed${SEED}forktech45p0/conditional_results.json"
    printf "  seed%-2d: Republic=%s  Gold=%s  MapMaking=%s\n" \
        $SEED \
        $([ -f "$REPUBLIC" ] && echo "✓" || echo "✗") \
        $([ -f "$GOLD" ] && echo "✓" || echo "✗") \
        $([ -f "$MAPMAKING" ] && echo "✓" || echo "✗")
done

log ""
log "Next steps:"
log "  1. Update setup_*_conditional_eval.py scripts to include seeds 11-20"
log "  2. Run: uv run python scripts/setup_republic_conditional_eval.py"
log "  3. Run: uv run python scripts/setup_gold500_conditional_eval.py"
log "  4. Run: uv run python scripts/setup_mapmaking_conditional_eval.py"
