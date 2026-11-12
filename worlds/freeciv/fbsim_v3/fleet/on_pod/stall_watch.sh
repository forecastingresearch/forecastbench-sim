#!/usr/bin/env bash
# stall_watch.sh ANCHOR RNG LOG [STALL_S=600] — stop the fork's run_fork.py once its game has gone STALL_S seconds without
# a new savegame (the client hangs after the server rejects a command), noting it in LOG; the caller then records the
# attempt as failed. Started next to fork_one.sh and killed when it returns.
A="$1"; R="$2"; LOG="$3"; STALL="${4:-600}"
D=/var/lib/tomcat10/webapps/data/savegames/seed${A}forkrng$R
PAT="[r]un_fork.py --base-seed $A .*rng_seed:$R( |\$)"
last=$(date +%s); step=$(( STALL < 60 ? 1 : 30 ))
while sleep "$step"; do
  now=$(date +%s); m=$(stat -c %Y "$D" 2>/dev/null || echo 0); [ "$m" -gt "$last" ] && last=$m
  pgrep -f "$PAT" > /dev/null || { last=$now; continue; }   # the game is not running (yet, or any more)
  if [ $((now - last)) -ge "$STALL" ]; then
    echo "stall_watch: no new savegame in $D for $((now - last)) s; stopping run_fork" >> "$LOG"
    pkill -TERM -f "$PAT"; exit 0
  fi
done
