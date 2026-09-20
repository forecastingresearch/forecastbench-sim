#!/usr/bin/env bash
# rerun_list.sh LIST_FILE [CONC=4] [TIMEOUT_S=5400] — run forks from LIST_FILE ("anchor rng" per line) on this pod
# with a long timeout, CONC at a time, publishing to /workspace/out/a<anchor>/forks/<name> exactly like the lanes.
# Use after the pod's queue has drained (lightly loaded pod => slow forks finish). Marks queue/done or queue/failed.
set -u
LIST="${1:?}"; CONC="${2:-4}"; TO="${3:-5400}"; F=/workspace/fleet; Q=$F/queue; B=/workspace/civbench; OUT=/workspace/out
mkdir -p $Q/done $Q/failed $OUT/logs
run1() { local A="$1" R="$2"; local t="a$A-rng$R"; local log="$OUT/logs/$t.rerun.log" t0 t1 rc
  rm -rf "$OUT/a$A/forks/rng$R" "$OUT/a$A/forks_failed/rng$R.FAILED" "$B/worlds/freeciv/logs/recordings/seed${A}forkrng$R" 2>/dev/null
  t0=$(date +%s); FBSIM_OUT_DIR="$OUT/a$A" FBSIM_FORK_TIMEOUT_S="$TO" timeout -k 60 $((TO+300)) bash "$B/fork_one.sh" "$A" "rng$R" "$R" 210 > "$log" 2>&1; rc=$?; t1=$(date +%s)
  if [ "$rc" -eq 0 ] && [ -s "$OUT/a$A/forks/rng$R/seed${A}forkrng${R}_data.json.gz" ]; then
    echo "$(date -u +%FT%TZ) rerun $((t1-t0))s $(ls $OUT/a$A/forks/rng$R/savegames | wc -l)saves" > "$Q/done/$t"; echo "DONE $t $((t1-t0))s"
  else { echo "rc=$rc secs=$((t1-t0)) rerun"; grep -m1 'reason:' "$OUT/a$A/forks_failed/rng$R.FAILED" 2>/dev/null; tail -c 4000 "$log"; } > "$Q/failed/$t.rerun"; echo "FAILED $t rc=$rc"; fi
}
export -f run1; export F Q B OUT TO
xargs -P "$CONC" -L 1 bash -c 'run1 $0 $1' < "$LIST"
echo "rerun_list done: $(wc -l < $LIST) tasks"
