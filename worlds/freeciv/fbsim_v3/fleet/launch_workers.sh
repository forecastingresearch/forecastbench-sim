#!/usr/bin/env bash
# launch_workers.sh ARCHIVE_ID POD_ID:LANES [POD_ID:LANES ...]
# For each worker pod: ship its queue slice (state/queue/<id>/todo), (re)place anchor saves, start the archive sync loop, start lanes.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ARCH="${1:?archive id}"; shift
read -r AIP APORT <<<"$(bash "$HERE/rp.sh" addr "$ARCH")"; [ -n "$APORT" ] || { echo "no archive addr"; exit 1; }
SSHKEY="$HOME/.ssh/id_ed25519_runpod"
for spec in "$@"; do
  id="${spec%%:*}"; lanes="${spec##*:}"
  read -r IP PORT <<<"$(bash "$HERE/rp.sh" addr "$id")"; [ -n "$PORT" ] || { echo "$id: no addr"; continue; }
  RSH="ssh -i $SSHKEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR"
  name="$(awk -F'\t' -v id="$id" '$2==id{print $3}' "$HERE/state/pods.tsv" | tail -1)"
  echo "== $id ($name) lanes=$lanes"
  rsync -az -e "$RSH" "$HERE/on_pod/" "root@$IP:/workspace/fleet/"
  $RSH "root@$IP" 'mkdir -p /workspace/fleet/queue/todo /workspace/fleet/queue/inprog /workspace/fleet/queue/done /workspace/fleet/queue/failed /workspace/fleet/logs'
  if [ -d "$HERE/state/queue/$id/todo" ]; then
    rsync -az --ignore-existing -e "$RSH" "$HERE/state/queue/$id/todo/" "root@$IP:/workspace/fleet/queue/todo/"
  fi
  for d in "$HERE"/state/anchors/seed*; do [ -d "$d/savegames" ] || continue; s=$(basename "$d")
    $RSH "root@$IP" "mkdir -p /workspace/civbench/worlds/freeciv/logs/recordings/$s/savegames"
    rsync -az -e "$RSH" "$d/savegames/" "root@$IP:/workspace/civbench/worlds/freeciv/logs/recordings/$s/savegames/"; done
  $RSH "root@$IP" "chmod -R a+rX /workspace/civbench/worlds/freeciv/logs/recordings 2>/dev/null; chmod 600 /workspace/fleet/archive_key 2>/dev/null
    pgrep -f 'fleet/sync_loop[.]sh' >/dev/null || nohup setsid bash /workspace/fleet/sync_loop.sh $AIP $APORT $name 300 >/dev/null 2>&1 </dev/null &
    sleep 1; bash /workspace/fleet/worker.sh $lanes 210; echo todo=\$(ls /workspace/fleet/queue/todo | wc -l) anchors=\$(ls /workspace/civbench/worlds/freeciv/logs/recordings/ | tr '\n' ' ')"
done
