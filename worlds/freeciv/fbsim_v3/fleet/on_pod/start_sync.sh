#!/usr/bin/env bash
# start_sync.sh ARCHIVE_IP ARCHIVE_PORT PODNAME [INTERVAL] — start the archive sync loop if not running.
if pgrep -f '/workspace/fleet/sync_loop.sh' >/dev/null; then echo "sync already running"; exit 0; fi
nohup setsid bash /workspace/fleet/sync_loop.sh "$1" "$2" "$3" "${4:-300}" >/dev/null 2>&1 </dev/null &
sleep 1; pgrep -af '/workspace/fleet/sync_loop.sh' | head -1 || echo "FAILED to start sync"
