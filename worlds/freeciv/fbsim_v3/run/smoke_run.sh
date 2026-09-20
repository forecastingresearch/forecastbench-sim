#!/usr/bin/env bash
# smoke_run.sh OUT_DIR [extra elicit args] — one elicit_v1.py process per model in models_v1.csv, each writing to
# OUT_DIR/<slug>/results.jsonl and OUT_DIR/logs/<slug>.log. Waits for all of them.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; OUT="$1"; shift
mkdir -p "$OUT/logs"; pids=""
while IFS=, read -r name eci id rest; do
  [ -n "$id" ] || continue
  slug="$(echo "$id" | tr '/:.' '___')"
  ( cd "$HERE" && uv run python elicit_v1.py --out "$OUT/$slug" --models "$id" "$@" > "$OUT/logs/$slug.log" 2>&1 ) </dev/null &
  pids="$pids $!"
done < <(tail -n +2 "$HERE/models_v1.csv")
wait $pids
echo "all model processes finished $(date -u +%FT%TZ)"
