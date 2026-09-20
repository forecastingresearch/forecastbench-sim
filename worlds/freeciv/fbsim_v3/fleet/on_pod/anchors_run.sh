#!/usr/bin/env bash
# anchors_run.sh MAX_TURNS SEED... — run anchor_one.sh for each seed concurrently (all at once), detached.
MAX_TURNS="${1:?}"; shift
mkdir -p /workspace/fleet/logs /workspace/out/anchors
for s in "$@"; do
  nohup setsid bash /workspace/fleet/anchor_one.sh "$s" "$MAX_TURNS" > "/workspace/fleet/logs/anchor_seed$s.log" 2>&1 < /dev/null &
  echo "seed$s pid $!"; sleep 2
done
