#!/usr/bin/env bash
# launch.sh OUT_DIR "id1,id2,..." [elicit args...] — one elicit_v1.py process per model id into OUT_DIR/<slug>/, logs in OUT_DIR/logs/
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; OUT="$1"; IDS="$2"; shift 2; mkdir -p "$OUT/logs"
for id in $(echo "$IDS" | tr ',' ' '); do
  slug="$(echo "$id" | tr '/:.' '___')"
  ( cd "$HERE" && nohup uv run python elicit_v1.py --out "$OUT/$slug" --models "$id" "$@" > "$OUT/logs/$slug.log" 2>&1 ) </dev/null &
done
echo "launched $(echo "$IDS" | tr ',' '\n' | wc -l | tr -d ' ') processes into $OUT"
