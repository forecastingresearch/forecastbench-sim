#!/usr/bin/env bash
# launch_workers.sh ARCHIVE_ID POD_ID:LANES [POD_ID:LANES ...]
# For each worker pod: ship its queue slice ($STATE/queue/<id>/todo), the run's run.env and the anchor saves, start the
# archive sync loop (to $VOL/<pod name>/) and the lanes (fork turn and end turn from run.env).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ARCH="${1:?archive id}"; shift
. "$HERE/paths.sh"
read -r AIP APORT <<<"$(bash "$HERE/rp.sh" addr "$ARCH")"; [ -n "$APORT" ] || { echo "no archive addr"; exit 1; }
for spec in "$@"; do
  id="${spec%%:*}"; lanes="${spec##*:}"
  read -r IP PORT <<<"$(bash "$HERE/rp.sh" addr "$id")"; [ -n "$PORT" ] || { echo "$id: no addr"; continue; }
  RSH="ssh -i '$SSHKEY' -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR"
  name="$(awk -F'\t' -v id="$id" '$2==id{print $3}' "$STATE/pods.tsv" | tail -1)"
  echo "== $id ($name) lanes=$lanes"
  rsync -az -e "$RSH" "$HERE/on_pod/" "root@$IP:/workspace/fleet/"
  rsync -az -e "$RSH" "$RUNDIR/run.env" "root@$IP:/workspace/fleet/run.env"
  podsh "$PORT" "root@$IP" 'mkdir -p /workspace/fleet/queue/todo /workspace/fleet/queue/inprog /workspace/fleet/queue/done /workspace/fleet/queue/failed /workspace/fleet/logs'
  if [ -d "$STATE/queue/$id/todo" ]; then
    rsync -az --ignore-existing -e "$RSH" "$STATE/queue/$id/todo/" "root@$IP:/workspace/fleet/queue/todo/"
  fi
  for d in "$STATE"/anchors/seed*; do [ -d "$d/savegames" ] || continue; s=$(basename "$d")
    podsh "$PORT" "root@$IP" "mkdir -p /workspace/civbench/data/freeciv/logs/recordings/$s/savegames"
    rsync -az -e "$RSH" "$d/savegames/" "root@$IP:/workspace/civbench/data/freeciv/logs/recordings/$s/savegames/"; done
  podsh "$PORT" "root@$IP" "chmod -R a+rX /workspace/civbench/data/freeciv/logs/recordings 2>/dev/null; chmod 600 /workspace/fleet/archive_key 2>/dev/null
    bash /workspace/fleet/start_sync.sh $AIP $APORT $FBSIM_RUN/$name 300   # a pgrep here would match this ssh command line itself
    bash /workspace/fleet/worker.sh $lanes; echo todo=\$(ls /workspace/fleet/queue/todo | wc -l) anchors=\$(ls /workspace/civbench/data/freeciv/logs/recordings/ | tr '\n' ' ')"
done
