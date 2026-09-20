#!/usr/bin/env bash
# worker.sh N_LANES [END_TURN=210] — start (or top up to) N lanes over the local queue.
# Queue: /workspace/fleet/queue/{todo,inprog,done,failed}; task file "a<anchor>-rng<rng>" containing "<anchor> <rng>".
# Claim = atomic mv todo -> inprog. Success -> done/<task> (timestamp host lane secs). Failure -> failed/<task>.<n>
# with the log tail; the task is re-queued once (second failure stays in failed/).
set -uo pipefail
if [ "${1:-}" = "__lane" ]; then N_LANES=0; else N_LANES="${1:?N_LANES}"; fi; END_TURN="${2:-210}"
F=/workspace/fleet; Q=$F/queue; B=/workspace/civbench; OUT=/workspace/out
mkdir -p $Q/todo $Q/inprog $Q/done $Q/failed $F/logs $OUT/logs
HOST="$(cat /tmp/f3_podid 2>/dev/null || hostname)"

lane() {
  local N="$1" idle_since=0 t A R rc t0 t1 attempt log
  exec >> "$F/logs/lane$N.log" 2>&1
  echo "$(date -u +%FT%TZ) lane$N start"
  sleep $(( (N * 3) % 20 ))
  while :; do
    t="$(ls $Q/todo 2>/dev/null | shuf -n1)"
    if [ -z "$t" ]; then
      [ "$idle_since" = 0 ] && idle_since=$(date +%s)
      if [ $(( $(date +%s) - idle_since )) -ge 900 ]; then echo "$(date -u +%FT%TZ) lane$N idle 15 min; exit"; return 0; fi
      sleep 30; continue
    fi
    idle_since=0
    mv "$Q/todo/$t" "$Q/inprog/$t.lane$N" 2>/dev/null || continue
    read -r A R < "$Q/inprog/$t.lane$N"
    attempt=$(( $(ls $Q/failed/ 2>/dev/null | grep -c "^$t\.") + 1 ))
    log="$OUT/logs/$t.a$attempt.log"
    # hygiene
    find /var/lib/tomcat10/webapps/data/savegames -type f -mmin +180 -delete 2>/dev/null
    rm -rf "$OUT/a$A/forks/rng$R" "$OUT/a$A/forks_failed/rng$R.FAILED" "$B/worlds/freeciv/logs/recordings/seed${A}forkrng$R" 2>/dev/null
    local free; free=$(df -m /workspace | awk 'NR==2{print $4}')
    if [ "${free:-0}" -lt 2500 ]; then echo "$(date -u +%FT%TZ) lane$N DISK LOW ${free}MB; pausing"; mv "$Q/inprog/$t.lane$N" "$Q/todo/$t"; sleep 300; continue; fi
    echo "$(date -u +%FT%TZ) lane$N START $t attempt $attempt"
    t0=$(date +%s)
    FBSIM_OUT_DIR="$OUT/a$A" FBSIM_FORK_TIMEOUT_S=1800 timeout -k 60 2100 bash "$B/fork_one.sh" "$A" "rng$R" "$R" "$END_TURN" > "$log" 2>&1
    rc=$?
    t1=$(date +%s)
    if [ "$rc" -eq 0 ] && [ -s "$OUT/a$A/forks/rng$R/seed${A}forkrng${R}_data.json.gz" ]; then
      echo "$(date -u +%FT%TZ) $HOST lane$N $((t1-t0))s attempt$attempt $(ls $OUT/a$A/forks/rng$R/savegames | wc -l)saves" > "$Q/done/$t"
      rm -f "$Q/inprog/$t.lane$N"
      echo "$(date -u +%FT%TZ) lane$N DONE $t in $((t1-t0))s"
      rm -f "$log"
    else
      { echo "rc=$rc secs=$((t1-t0)) host=$HOST lane=$N"; grep -m1 'reason:' "$OUT/a$A/forks_failed/rng$R.FAILED" 2>/dev/null; echo '--- log tail ---'; tail -c 6000 "$log" 2>/dev/null; } > "$Q/failed/$t.$attempt"
      rm -f "$Q/inprog/$t.lane$N"
      rm -rf "$OUT/a$A/forks/rng$R" "$B/worlds/freeciv/logs/recordings/seed${A}forkrng$R" 2>/dev/null
      echo "$(date -u +%FT%TZ) lane$N FAILED $t attempt $attempt rc=$rc ($((t1-t0))s)"
      if [ "$attempt" -lt 2 ]; then printf '%s %s\n' "$A" "$R" > "$Q/todo/$t"; fi
    fi
  done
}

if [ "${1:-}" = "__lane" ]; then END_TURN="${3:-210}"; lane "$2"; exit 0; fi

# top up lanes to N_LANES (never kill running ones)
alive=0; : > $F/lanes.pids.new
if [ -f $F/lanes.pids ]; then while read -r n p; do kill -0 "$p" 2>/dev/null && { echo "$n $p" >> $F/lanes.pids.new; alive=$((alive+1)); }; done < $F/lanes.pids; fi
mv $F/lanes.pids.new $F/lanes.pids
used="$(awk '{print $1}' $F/lanes.pids)"
i=0; started=0
while [ $((alive + started)) -lt "$N_LANES" ]; do
  if ! grep -qx "$i" <<<"$used"; then
    nohup setsid bash "$0" __lane "$i" "$END_TURN" > /dev/null 2>&1 < /dev/null &
    echo "$i $!" >> $F/lanes.pids; started=$((started+1))
  fi
  i=$((i+1))
done
echo "lanes alive=$alive started=$started total=$((alive+started))"
