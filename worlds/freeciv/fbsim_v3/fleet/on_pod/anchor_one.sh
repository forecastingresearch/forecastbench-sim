#!/usr/bin/env bash
# anchor_one.sh SEED [MAX_TURNS=fork turn + 1] — play a fresh all-AI world to MAX_TURNS, serialize with savegame
# coverage, run the observability + old publish gates AS REPORTS (never blocking), publish data + all
# per-turn saves, ruleset.json, sgtables and the recording's turn states (for regen_reports.py) to
# /workspace/out/anchors/seed<SEED>/. The recording dir is KEPT (forks read its fork-turn save).
# Settings from /workspace/fleet/run.env (bringup.sh ships the run's): FBSIM_N_PLAYERS (default 5), FBSIM_BARBARIANS
# (e.g. DISABLED; default: server default), FBSIM_FORK_TURN (default 60).
set -uo pipefail
[ -f /workspace/fleet/run.env ] && . /workspace/fleet/run.env
FORK="${FBSIM_FORK_TURN:-60}"
SEED="${1:?SEED}"; MAX_TURNS="${2:-$((FORK + 1))}"; NP="${FBSIM_N_PLAYERS:-5}"; BARB="${FBSIM_BARBARIANS:-}"
B=/workspace/civbench; source $B/pod_env.sh
export FBSIM_OUT_DIR=/workspace/out/anchors_work
REC="$FBSIM_LOG_ROOT/recordings/seed$SEED"; OUT="/workspace/out/anchors/seed$SEED"; SER="$FBSIM_OUT_DIR/ser_seed$SEED"
mkdir -p "$OUT" "$SER"
printf "players=%s\nbarbarians=%s\nfork_turn=%s\nrun=%s\n" "$NP" "${BARB:-default}" "$FORK" "${FBSIM_RUN:-}" > "$OUT/settings.txt"
echo "== seed$SEED players=$NP barbarians=${BARB:-default} run_world $MAX_TURNS turns $(date -u +%FT%TZ)"
timeout -k 60 3600 python $B/worlds/freeciv/scripts/run_world.py --seed "$SEED" --max_turns "$MAX_TURNS" \
  --num_ai_players "$NP" ${BARB:+--barbarians "$BARB"} --quiet; rc=$?
echo "run_world rc=$rc $(date -u +%FT%TZ)"; echo "$rc" > "$OUT/run_world.rc"
N=$(ls "$REC/savegames" 2>/dev/null | grep -c "seed${SEED}_T" || true); echo "savegames: $N"
export FBSIM_REQUIRE_SAVEGAME_COVERAGE=1
python $B/worlds/freeciv/scripts/generate_data_batch.py --input-dir "$FBSIM_LOG_ROOT/recordings" --output-dir "$SER" --filter "seed$SEED" --force --workers 1 > "$OUT/serialize.log" 2>&1; echo "serialize rc=$?"
D="$SER/seed${SEED}_data.json"
[ -s "$D" ] || { echo "NO DATA JSON"; exit 1; }
python $B/bench/observability_gate.py "$D" --recording-dir "$REC" --username "seed$SEED" > "$OUT/obsgate.txt" 2>&1; echo "obsgate rc=$? (report only)"
if [ "$NP" -gt 1 ]; then python $B/bench/world_publish_gate.py "$D" > "$OUT/publishgate.txt" 2>&1; echo "publishgate rc=$? (report only)"
else echo "publishgate skipped: one player has no diplomacy" | tee "$OUT/publishgate.txt"; fi
gzip -c "$D" > "$OUT/seed${SEED}_data.json.gz"
mkdir -p "$OUT/savegames"; cp "$REC/savegames/seed${SEED}_T"*.sav.* "$OUT/savegames/"
cp "$REC/ruleset.json" "$OUT/" 2>/dev/null || echo "no ruleset.json in $REC"
tar -czf "$OUT/recording_states.tar.gz" -C "$REC" $(cd "$REC" && ls | grep "^turn_.*_state[.]json$")
python $B/sg_tables.py "$OUT" --workers 4 > "$OUT/sg_tables.log" 2>&1; echo "sg_tables rc=$?"
echo "published $(ls $OUT/savegames | wc -l) saves -> $OUT"
rm -rf "$SER"
