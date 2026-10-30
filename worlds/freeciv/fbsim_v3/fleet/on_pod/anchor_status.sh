#!/usr/bin/env bash
# anchor_status.sh SEED... — one line "SEED done|running|failed|pending" per seed, from anchor_one.sh's processes and
# logs. The process is checked first: once it has exited, its log is final.
for s in "$@"; do
  L="/workspace/fleet/logs/anchor_seed$s.log"
  if pgrep -f "anchor_one[.]sh $s\$" >/dev/null; then echo "$s running"
  elif grep -q '^published ' "$L" 2>/dev/null; then echo "$s done"
  elif [ -f "$L" ]; then echo "$s failed"
  else echo "$s pending"; fi
done
