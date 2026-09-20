#!/bin/bash
# launch_effort_v2.sh OUT ENVFILE PYTHON — one process per model for the grouped-prompt effort check, 2 s apart; waits for all.
OUT="$1"; ENVF="$2"; PY="$3"; HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"
for m in qwen/qwen3-235b-a22b anthropic/claude-haiku-4.5 openai/o3 openai/gpt-5 anthropic/claude-sonnet-5 openai/gpt-5.6-luna anthropic/claude-fable-5; do
  slug="${m//\//_}"
  nohup "$PY" "$HERE/effort_check_v2.py" --out "$OUT/$slug" --models "$m" --env-file "$ENVF" > "$OUT/$slug.log" 2>&1 &
  perl -e 'select(undef,undef,undef,2)'
done
wait
echo "all models finished $(date -u +%FT%TZ)"
for f in "$OUT"/*.log; do echo "--- $f"; tail -2 "$f"; done
