#!/usr/bin/env bash
# finish_all.sh [--dry-run] — end-of-run automation, idempotent, safe to re-run every few minutes:
#  1. worker with todo=0 & inprog=0 & no FINISHED & finish not running  -> start finish_pod.sh (sg_tables + final sync)
#  2. worker with FINISHED and archive copy complete (fork dirs, data files, sgtables all match the pod) -> delete pod
# Never touches the archive pods ($STATE/archive_pods.txt; the first one is checked for the archive copy).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"; DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
. "$HERE/paths.sh"
ARCH="$(grep -v '^#' "$STATE"/archive_pods.txt | grep . | head -n 1)"; [ -n "$ARCH" ] || { echo "no archive pod in $STATE/archive_pods.txt"; exit 1; }
read -r AIP APORT <<<"$(bash rp.sh addr "$ARCH")"
ARSH() { podsh "$APORT" "root@$AIP" "$@"; }
while IFS=$'\t' read -r ts id name flavor vcpu cloud dc cost status; do
  case "$ts" in \#*) continue;; esac; [ "$status" = deleted ] && continue
  grep -qx "$id" "$STATE"/archive_pods.txt 2>/dev/null && continue
  grep -qx "$id" "$STATE"/keep.txt 2>/dev/null && { echo "$id $name: kept ($STATE/keep.txt)"; continue; }
  st="$(bash rp.sh ssh "$id" 'F=/workspace/fleet; echo todo=$(ls $F/queue/todo | wc -l) inprog=$(ls $F/queue/inprog | wc -l) done=$(ls $F/queue/done | wc -l) finished=$([ -f $F/FINISHED ] && echo 1 || echo 0) finrun=$(pgrep -fc "fleet/finish_pod[.]sh") forks=$(ls -d /workspace/out/a*/forks/rng* 2>/dev/null | wc -l) sgt=$(ls /workspace/out/a*/forks/rng*/*_sgtables.json.gz 2>/dev/null | wc -l) trunc=$(ls /workspace/out/a*/forks/rng*/TRUNCATED 2>/dev/null | wc -l)' 2>/dev/null </dev/null)" || { echo "$id $name UNREACHABLE"; continue; }
  eval "$st"
  if [ "$todo" = 0 ] && [ "$inprog" = 0 ] && [ "$finished" = 0 ] && [ "$finrun" = 0 ]; then
    echo "$id $name: drained ($done done, $forks forks) -> START finish_pod"
    [ $DRY = 1 ] || bash rp.sh ssh "$id" 'nohup setsid bash /workspace/fleet/finish_pod.sh > /workspace/fleet/logs/finish.log 2>&1 < /dev/null &' </dev/null
  elif [ "$finished" = 1 ]; then
    a="$(ARSH "echo \$(ls -d $VOL/$name/a*/forks/rng* 2>/dev/null | wc -l) \$(ls $VOL/$name/a*/forks/rng*/*_data.json.gz 2>/dev/null | wc -l) \$(ls $VOL/$name/a*/forks/rng*/*_sgtables.json.gz 2>/dev/null | wc -l)" </dev/null)"
    read -r af ad as <<<"$a"
    if [ "$af" = "$forks" ] && [ "$ad" = "$forks" ] && [ "$as" = "$forks" ] && [ "$forks" -gt 0 ]; then
      echo "$id $name: FINISHED, archive complete ($af forks, $as sgtables) -> DELETE"
      [ $DRY = 1 ] || bash rp.sh delete "$id" </dev/null
    else
      echo "$id $name: FINISHED but archive has forks=$af data=$ad sgtables=$as vs pod forks=$forks sgt=$sgt -> wait"
    fi
  else
    echo "$id $name: todo=$todo inprog=$inprog done=$done forks=$forks sgt=$sgt trunc=$trunc finrun=$finrun"
  fi
done < "$STATE"/pods.tsv
