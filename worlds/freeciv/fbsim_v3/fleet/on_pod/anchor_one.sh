#!/usr/bin/env bash
# anchor_one.sh SEED [MAX_TURNS=61] — play a fresh all-AI world to MAX_TURNS, serialize with savegame
# coverage, run the observability + old publish gates AS REPORTS (never blocking), publish data + all
# per-turn saves to /workspace/out/anchors/seed<SEED>/. The recording dir is KEPT (forks read its T60 save).
set -uo pipefail
SEED="${1:?SEED}"; MAX_TURNS="${2:-61}"
B=/workspace/civbench; source $B/pod_env.sh
export FBSIM_OUT_DIR=/workspace/out/anchors_work
REC="$FBSIM_LOG_ROOT/recordings/seed$SEED"; OUT="/workspace/out/anchors/seed$SEED"; SER="$FBSIM_OUT_DIR/ser_seed$SEED"
mkdir -p "$OUT" "$SER"
echo "== seed$SEED run_world $MAX_TURNS turns $(date -u +%FT%TZ)"
timeout -k 60 3600 python $B/worlds/freeciv/scripts/run_world.py --seed "$SEED" --max_turns "$MAX_TURNS" --quiet; rc=$?
echo "run_world rc=$rc $(date -u +%FT%TZ)"; echo "$rc" > "$OUT/run_world.rc"
N=$(ls "$REC/savegames" 2>/dev/null | grep -c "seed${SEED}_T" || true); echo "savegames: $N"
export FBSIM_REQUIRE_SAVEGAME_COVERAGE=1
python $B/worlds/freeciv/scripts/generate_data_batch.py --input-dir "$FBSIM_LOG_ROOT/recordings" --output-dir "$SER" --filter "seed$SEED" --force --workers 1 > "$OUT/serialize.log" 2>&1; echo "serialize rc=$?"
D="$SER/seed${SEED}_data.json"
[ -s "$D" ] || { echo "NO DATA JSON"; exit 1; }
python $B/bench/observability_gate.py "$D" --recording-dir "$REC" --username "seed$SEED" > "$OUT/obsgate.txt" 2>&1; echo "obsgate rc=$? (report only)"
python $B/bench/world_publish_gate.py "$D" > "$OUT/publishgate.txt" 2>&1; echo "publishgate rc=$? (report only)"
gzip -c "$D" > "$OUT/seed${SEED}_data.json.gz"
mkdir -p "$OUT/savegames"; cp "$REC/savegames/seed${SEED}_T"*.sav.* "$OUT/savegames/"
echo "published $(ls $OUT/savegames | wc -l) saves -> $OUT"
rm -rf "$SER"
