#!/usr/bin/env bash
# bringup.sh POD_ID — wait for ssh, verify deadman + services, ship bundle + on_pod scripts + the run's run.env, pod_setup,
# place anchors.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/paths.sh"
ID="${1:?POD_ID}"
BUNDLE="${FBSIM_POD_BUNDLE:-$STATE/bundle}"   # the bundle rsynced to every pod (no .venv; pod_setup.sh builds it on the pod)
[ -n "${FBSIM_POD_BUNDLE:-}" ] || bash "$HERE/make_bundle.sh" "$BUNDLE"   # rebuilt each time so code changes reach new pods; FBSIM_POD_BUNDLE pins one as is
for i in $(seq 1 60); do a="$(bash "$HERE/rp.sh" addr "$ID")"; [ -n "$a" ] && break; sleep 15; done
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
RSH="ssh -i '$SSHKEY' -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR"
rsync -az --delete --exclude .venv --exclude uv.lock --exclude __pycache__ --exclude '*.pyc' -e "$RSH" "$BUNDLE/" "root@$IP:/workspace/civbench/" || exit 1
rsync -az -e "$RSH" "$HERE/on_pod/" "root@$IP:/workspace/fleet/" || exit 1
rsync -az -e "$RSH" "$RUNDIR/run.env" "root@$IP:/workspace/fleet/run.env" || exit 1   # the pod scripts source it
[ -f "$ARCHKEY" ] && rsync -az -e "$RSH" "$ARCHKEY" "root@$IP:/workspace/fleet/archive_key" && S 'chmod 600 /workspace/fleet/archive_key'
echo "== pod_setup"
# detached, with its exit code in a file, so a dropped ssh session cannot fail a healthy pod; polled for up to 40 min
S 'rm -f /workspace/fleet/logs/pod_setup.rc; cd /workspace/civbench && nohup setsid bash -c "bash pod_setup.sh > /workspace/fleet/logs/pod_setup.log 2>&1; echo \$? > /workspace/fleet/logs/pod_setup.rc" < /dev/null > /dev/null 2>&1 &'
rc=""; for i in $(seq 1 160); do rc="$(S 'cat /workspace/fleet/logs/pod_setup.rc 2>/dev/null' 2>/dev/null)"; [ -n "$rc" ] && break; sleep 15; done
S 'tail -3 /workspace/fleet/logs/pod_setup.log'; echo "pod_setup_rc=${rc:-timeout}"
if ls "$STATE/anchors"/seed*/savegames/*_T${FBSIM_FORK_TURN}_* >/dev/null 2>&1; then
  echo "== place anchors"
  for d in "$STATE"/anchors/seed*; do s=$(basename "$d"); S "mkdir -p /workspace/civbench/data/freeciv/logs/recordings/$s/savegames"; rsync -az -e "$RSH" "$d/savegames/" "root@$IP:/workspace/civbench/data/freeciv/logs/recordings/$s/savegames/"; done
  S 'chmod -R a+rX /workspace/civbench/data/freeciv/logs/recordings; ls /workspace/civbench/data/freeciv/logs/recordings/'
fi
echo "bringup done $ID $IP:$PORT"
