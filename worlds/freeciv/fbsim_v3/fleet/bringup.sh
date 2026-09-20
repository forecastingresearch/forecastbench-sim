#!/usr/bin/env bash
# bringup.sh POD_ID — wait for ssh, verify deadman + services, ship bundle + on_pod scripts, pod_setup, place anchors.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ID="${1:?POD_ID}"
BUNDLE=/Users/jaeholee0404/civbench/tmp/pilot_v2/fleet_prep/pod_bundle
SSHKEY="$HOME/.ssh/id_ed25519_runpod"
for i in $(seq 1 60); do a="$(bash $HERE/rp.sh addr "$ID")"; [ -n "$a" ] && break; sleep 15; done
[ -n "${a:-}" ] || { echo "no address after 15 min"; exit 1; }
read -r IP PORT <<<"$a"; echo "addr $IP:$PORT"
S() { ssh -i "$SSHKEY" -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes -o LogLevel=ERROR "root@$IP" "$@"; }
for i in $(seq 1 60); do S true 2>/dev/null && break; sleep 15; done
S true || { echo "ssh never came up"; exit 1; }
echo "== deadman + services"
S 'pgrep -af "fleet/deadman[.]sh" | head -1; test -s /tmp/f3_apikey && echo apikey_staged; nproc; free -g | sed -n 2p; df -h /workspace | tail -1'
for i in $(seq 1 60); do
  if S 'curl -fsS -m 5 -o /dev/null http://localhost:80/ && [ "$(pgrep -c -f "freeciv-web --debug")" -ge 1 ]' 2>/dev/null; then break; fi; sleep 10
done
S 'echo "civservers=$(pgrep -c -f "freeciv-web --debug")"; pgrep -f "[t]omcat" >/dev/null && echo tomcat_up || echo tomcat_DOWN; pgrep -f "[m]ysqld|[m]ariadbd" >/dev/null && echo mysql_up || echo mysql_DOWN'
echo "== ship bundle + scripts"
RSH="ssh -i $SSHKEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR"
rsync -az --delete --exclude .venv --exclude uv.lock --exclude __pycache__ --exclude '*.pyc' -e "$RSH" "$BUNDLE/" "root@$IP:/workspace/civbench/" || exit 1
rsync -az -e "$RSH" "$HERE/on_pod/" "root@$IP:/workspace/fleet/" || exit 1
[ -f "$HERE/state/archive_key" ] && rsync -az -e "$RSH" "$HERE/state/archive_key" "root@$IP:/workspace/fleet/archive_key" && S 'chmod 600 /workspace/fleet/archive_key'
echo "== pod_setup"
S 'cd /workspace/civbench && bash pod_setup.sh > /workspace/fleet/logs/pod_setup.log 2>&1; rc=$?; tail -3 /workspace/fleet/logs/pod_setup.log; echo pod_setup_rc=$rc'
if ls "$HERE/state/anchors"/seed*/savegames/*_T60_* >/dev/null 2>&1; then
  echo "== place anchors"
  for d in "$HERE"/state/anchors/seed*; do s=$(basename "$d"); S "mkdir -p /workspace/civbench/worlds/freeciv/logs/recordings/$s/savegames"; rsync -az -e "$RSH" "$d/savegames/" "root@$IP:/workspace/civbench/worlds/freeciv/logs/recordings/$s/savegames/"; done
  S 'chmod -R a+rX /workspace/civbench/worlds/freeciv/logs/recordings; ls /workspace/civbench/worlds/freeciv/logs/recordings/'
fi
echo "bringup done $ID $IP:$PORT"
