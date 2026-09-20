#!/bin/bash
# launch_run2.sh RUN_NAME ENV_FILE — run 2 in full: the batched arm for all 24 models and the unbatched natural-conditional
# arm for the two re-pinned models, into ../results/RUN_NAME/.  Seeds the batched folder with the smoke's completed calls
# (the same batches of the same design, so they are simply not re-asked) and the natcond folder with run 1's rows for the
# other 22 models.  Idempotent: rerunning resumes.
#   bash launch_run2.sh run2_2026-09-20 ~/path/to/.env
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; RUN="$HERE/../results/$1"; ENVF="$2"
mkdir -p "$RUN/batched" "$RUN/natcond"
if [ -d "$HERE/../results/run2_smoke/batched" ]; then
  for d in "$HERE"/../results/run2_smoke/batched/*/; do s=$(basename "$d"); [ "$s" = logs ] && continue
    mkdir -p "$RUN/batched/$s"; for f in calls.jsonl results.jsonl; do [ -f "$d/$f" ] && [ ! -f "$RUN/batched/$s/$f" ] && cp "$d/$f" "$RUN/batched/$s/$f"; done; done
  echo "seeded $RUN/batched with the smoke's completed calls"
fi
[ -d "$RUN/natcond/from_run1" ] || cp -R "$HERE/../results/run2_natcond/from_run1" "$RUN/natcond/from_run1"
T0=$(date +%s)
( cd "$HERE" && python3 elicit_natcond_v1.py --models "DeepSeek: DeepSeek V3,DeepSeek: DeepSeek V4 Flash 0731" --out "$RUN/natcond/repinned" --workers 16 --env-file "$ENVF" > "$RUN/natcond_repinned.log" 2>&1 ) &
NCPID=$!
bash "$HERE/launch_v2.sh" "$RUN/batched" "$ENVF"
wait $NCPID; echo "natcond rerun finished; $(tail -n 2 "$RUN/natcond_repinned.log" | cut -c1-160)"
echo "run 2 elicitation finished after $(( $(date +%s) - T0 )) s"
