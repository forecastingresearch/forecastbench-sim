#!/usr/bin/env bash
# 5_finish.sh RUN [--interval SECONDS=300] — step 5: wait for every worker to drain its queue (printing the fleet status
# every interval), list the tasks that failed twice and offer to rerun them on the drained workers (rerun_list.sh,
# 90-minute timeout), then finish the workers (exact savegame tables, final archive sync) and delete each one once the
# archive holds all its forks (finish_all.sh). Safe to stop and rerun. Next: 6_pull_corpus.sh RUN
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/steps_lib.sh"
step_init "${1:-}"; shift; IV=300
while [ $# -gt 0 ]; do case "$1" in --interval) IV="$2"; shift 2;; *) die "bad option $1";; esac; done
[ -n "$(live_workers)" ] || { say "no live workers; next: ./6_pull_corpus.sh $FBSIM_RUN"; exit 0; }
# pod_q ID -> "todo inprog finished" on that worker, or nothing if unreachable
pod_q() { ./rp.sh ssh "$1" 'F=/workspace/fleet; echo $(ls $F/queue/todo | wc -l) $(ls $F/queue/inprog | wc -l) $([ -f $F/FINISHED ] && echo 1 || echo 0)' 2>/dev/null < /dev/null; }

# ---- 1. wait until every worker is drained
while :; do
  busy=0; started_finish=0
  for id in $(live_workers); do
    read -r todo inprog fin <<<"$(pod_q "$id")"
    [ -n "${todo:-}" ] || { busy=1; say "worker $id unreachable"; continue; }
    [ "$fin" = 1 ] && started_finish=1
    [ "$todo" = 0 ] && [ "$inprog" = 0 ] || busy=1
  done
  [ "$busy" = 0 ] && break
  ./status.sh | sed 's/^/    /'
  say "workers still busy; next check in $IV s"
  sleep "$IV"
done
say "every worker is drained"

# ---- 2. tasks that failed twice
if [ ! -f "$S/.exhausted_handled" ] && [ "$started_finish" = 0 ]; then
  say "looking for tasks that failed twice"
  ./exhausted.sh | sed 's/^/    /'
  n=$(grep -c . "$S/exhausted.txt" 2>/dev/null || true)
  if [ "${n:-0}" -gt 0 ] && confirm "rerun these $n tasks on the drained workers (90-minute timeout each) before finishing?"; then
    W=($(live_workers)); i=0; rm -rf "$S/rerun"; mkdir -p "$S/rerun"
    while read -r a r; do echo "$a $r" >> "$S/rerun/${W[$((i % ${#W[@]}))]}.txt"; i=$((i + 1)); done < "$S/exhausted.txt"
    for f in "$S"/rerun/*.txt; do
      id="$(basename "$f" .txt)"; read -r IP PORT <<<"$(./rp.sh addr "$id")"
      rsync -az -e "ssh -i '$SSHKEY' -p $PORT ${SSHO[*]}" "$f" "root@$IP:/workspace/fleet/rerun.txt" \
        && ./rp.sh ssh "$id" 'rm -f /workspace/fleet/logs/rerun.log; nohup setsid bash /workspace/fleet/rerun_list.sh /workspace/fleet/rerun.txt 4 5400 > /workspace/fleet/logs/rerun.log 2>&1 < /dev/null &' < /dev/null \
        && say "rerunning $(wc -l < "$f" | tr -d ' ') tasks on $id"
    done
    while :; do
      left=0
      for f in "$S"/rerun/*.txt; do
        id="$(basename "$f" .txt)"
        ./rp.sh ssh "$id" 'grep -q "^rerun_list done" /workspace/fleet/logs/rerun.log' < /dev/null 2>/dev/null || left=$((left + 1))
      done
      [ "$left" = 0 ] && break
      say "reruns still going on $left workers; next check in $IV s"; sleep "$IV"
    done
    say "reruns done; tasks still failed:"; ./exhausted.sh | sed 's/^/    /'
  fi
  touch "$S/.exhausted_handled"
fi

# ---- 3. finish and delete
while [ -n "$(live_workers)" ]; do
  say "finish_all.sh: finishing drained workers, deleting finished ones whose forks are all archived"
  ./finish_all.sh | sed 's/^/    /'
  [ -n "$(live_workers)" ] || break
  say "$(live_workers | wc -l | tr -d ' ') workers left; next check in 120 s"; sleep 120
done
say "all workers are finished and deleted. Next: ./6_pull_corpus.sh $FBSIM_RUN"
