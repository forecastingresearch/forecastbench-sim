#!/usr/bin/env bash
# launch_pending.sh [--max-retries N=5] — launch lanes on any worker pod whose bringup finished but has no launch log
# yet, or whose last launch ended without starting lanes (that log is kept as failed_launch_<id>_<time>.log and an ERROR
# is printed). A worker whose launch failed more than N times is not retried; the script then exits 2.
# Workers sync to the archive pods in $STATE/archive_pods.txt (one ID per line), dealt round-robin.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
. "$HERE/paths.sh"
MAX_RETRIES=5
while [ $# -gt 0 ]; do case "$1" in --max-retries) MAX_RETRIES="$2"; shift 2;; *) echo "bad option $1"; exit 1;; esac; done
gave_up=0
i=0; ARCHS=($(grep -v '^#' "$STATE"/archive_pods.txt | grep .)); [ ${#ARCHS[@]} -gt 0 ] || { echo "no archive pods in $STATE/archive_pods.txt"; exit 1; }
while IFS=$'\t' read -r ts id name flavor vcpu cloud dc cost status; do
  case "$ts" in \#*) continue;; esac
  [ "$status" = deleted ] && continue
  grep -qx "$id" "$STATE"/archive_pods.txt 2>/dev/null && continue
  log="$STATE/launch_$id.log"
  if [ -f "$log" ]; then
    grep -q "^launch_rc=" "$log" && ! grep -q "lanes alive=" "$log" || continue   # running, or lanes started
    kept="$STATE/failed_launch_${id}_$(date +%H%M%S).log"; mv "$log" "$kept"
    echo "ERROR: launch on $id ($name) failed: $(grep -v "^launch_rc=" "$kept" | tail -n 1 | cut -c1-80) (log: $kept)"
  fi
  nfail=$(ls "$STATE"/failed_launch_"$id"*.log 2>/dev/null | wc -l | tr -d ' ')
  if [ "$nfail" -gt "$MAX_RETRIES" ]; then
    echo "ERROR: launch on $id ($name) failed $nfail times; not retrying (--max-retries $MAX_RETRIES)"; gave_up=1; continue
  fi
  [ -d "$STATE/queue/$id/todo" ] || { echo "$id: no queue slice"; continue; }
  if grep -q "bringup done" "$STATE/bringup_$id.log" 2>/dev/null && grep -q "pod_setup_rc=0" "$STATE/bringup_$id.log"; then
    arch=${ARCHS[$((i % ${#ARCHS[@]}))]}; i=$((i+1))
    lanes=$((vcpu*3/4))
    nohup bash -c 'bash launch_workers.sh "$@"; echo "launch_rc=$?"' _ $arch $id:$lanes > "$log" 2>&1 &   # launch_rc= marks the end
    echo "launched $id ($name) lanes=$lanes -> $arch$([ "$nfail" -gt 0 ] && echo " (retry $nfail of $MAX_RETRIES)")"
  else echo "pending $id ($name): $(tail -n 1 "$STATE"/bringup_$id.log 2>/dev/null | cut -c1-60)"; fi
done < "$STATE"/pods.tsv
[ "$gave_up" = 0 ] || exit 2
