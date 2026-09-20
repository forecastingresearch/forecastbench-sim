#!/usr/bin/env bash
# sync_loop.sh ARCHIVE_IP ARCHIVE_PORT PODNAME [INTERVAL=600] — push /workspace/out to the archive pod forever.
set -u
IP="$1"; PORT="$2"; NAME="$3"; IV="${4:-600}"
F=/workspace/fleet
while :; do
  rsync -a --partial --timeout=120 -e "ssh -i $F/archive_key -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR" \
    /workspace/out/ "root@$IP:/vol/fbsim_v3/$NAME/" >> $F/logs/sync.log 2>&1
  echo "$(date -u +%FT%TZ) sync rc=$?" >> $F/logs/sync.log
  sleep "$IV"
done
