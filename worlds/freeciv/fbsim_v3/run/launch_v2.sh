#!/bin/bash
# launch_v2.sh OUT_DIR ENV_FILE [elicit_v2 args...] — one elicit_v2.py process per model in models_v2.csv, started two
# seconds apart, into OUT_DIR/<slug>/ with a log in OUT_DIR/logs/<slug>.log.  Waits for all of them and prints each
# model's last log line.  Resumable: a rerun skips calls already stored in calls.jsonl.
#   bash launch_v2.sh ../results/run2_smoke/batched ~/path/to/.env --smoke
#   bash launch_v2.sh ../results/run2_batched ~/path/to/.env
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; OUT="$1"; ENVF="$2"; shift 2
mkdir -p "$OUT/logs"; pids=(); T0=$(date +%s)
while IFS=, read -r name eci id rest; do
  [ -n "$id" ] && [ "$id" != "openrouter_id" ] || continue
  slug="$(echo "$id" | tr '/:.' '___')"
  ( cd "$HERE" && python3 elicit_v2.py --out "$OUT/$slug" --models "$id" --env-file "$ENVF" "$@" > "$OUT/logs/$slug.log" 2>&1 ) &
  pids+=($!); sleep 2
done < "$HERE/models_v2.csv"
echo "launched ${#pids[@]} model processes at $(date -u +%H:%M:%SZ) into $OUT"
for p in "${pids[@]}"; do wait "$p"; done
echo "all finished after $(( $(date +%s) - T0 )) s"
for f in "$OUT"/logs/*.log; do echo "$(basename "$f" .log): $(grep -E '^[A-Za-z].*rows|calls,' "$f" | tail -n1 | cut -c1-160)"; done
