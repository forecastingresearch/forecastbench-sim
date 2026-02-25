#!/bin/bash
# Run Map Making (tech 45) forks for all seeds where it's researchable.
# Must run each fork in a separate process due to civrealm global singleton state.
#
# Seeds with Map Making researchable at turn 60: 0,1,2,4,5,6,7,8,9,10
# Seed 3 does NOT have Map Making researchable — skipped.
#
# Usage:
#   cd ~/Projects/civbench
#   bash scripts/run_mapmaking_forks.sh

set -e

SEEDS=(0 1 2 4 5 6 7 8 9 10)
CHECKPOINT=60
END_TURN=150
TECH_ID=45  # Map Making

echo "=== Running Map Making (tech ${TECH_ID}) forks ==="
echo "Seeds: ${SEEDS[*]}"
echo "Checkpoint: turn ${CHECKPOINT}, End: turn ${END_TURN}"
echo ""

for SEED in "${SEEDS[@]}"; do
    FORK_DIR="logs/recordings/seed${SEED}forktech${TECH_ID}p0"

    if [ -d "$FORK_DIR" ] && [ -f "$FORK_DIR/conditional_results.json" ]; then
        echo "[seed${SEED}] Fork already exists with conditional_results.json — skipping"
        continue
    fi

    echo "[seed${SEED}] Running fork..."
    uv run python scripts/run_fork.py \
        --base-seed "$SEED" \
        --checkpoint-turn "$CHECKPOINT" \
        --modification "tech:0:${TECH_ID}" \
        --end-turn "$END_TURN" \
        --recording-dir "logs/recordings/seed${SEED}" \
        --fork-name "tech${TECH_ID}p0"

    echo "[seed${SEED}] Generating conditional results..."
    uv run python scripts/generate_conditional_results.py \
        --baseline-dir "logs/recordings/seed${SEED}" \
        --fork-dir "$FORK_DIR" \
        --condition "tech:0:${TECH_ID}" \
        --checkpoint-turn "$CHECKPOINT" \
        --end-turn "$END_TURN" \
        --game-data "data/games/seed${SEED}_data.json" \
        --verbose

    echo "[seed${SEED}] Done"
    echo ""
done

echo "=== All forks complete ==="
echo "Next: run setup_mapmaking_conditional_eval.py to create evaluation directories"
