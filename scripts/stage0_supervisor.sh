#!/bin/bash
# Overnight supervisor for the test-world labeling fleet (seeds 348-359).
# - relaunches any worker whose seed range is incomplete but has no process
# - wipes partial artifacts of a seed before relaunch (bad mc json, or a
#   recording whose questions.json came out empty)
# - sheds worker E if swap exceeds 18GB; re-adds when it drops below 13GB
# Emits one line per action (consumed by a Monitor).
cd /Users/jaeholee0404/civbench

declare -A RANGES=( [C2]="348 349 350 351" [D]="352 353 354 355" [E]="356 357 358 359" )

seed_done() {
  python3 -c "
import json,sys
try:
    d=json.load(open('tmp/mc_resolve/seed$1_h1.json'))
    nr=d.get('n_rollouts',{})
    ok=bool(d.get('p_mc')) and nr and min(nr.values())>=30
    print(1 if ok else 0)
except Exception: print(0)" 2>/dev/null
}

clean_seed() {
  s=$1
  # wipe mc json if invalid
  if [ -f "tmp/mc_resolve/seed${s}_h1.json" ] && [ "$(seed_done $s)" != "1" ]; then
    rm -f "tmp/mc_resolve/seed${s}_h1.json"; echo "cleaned partial mc seed$s"
  fi
  # wipe recording if questions exist but are empty (partial world run)
  q="data/questions_mc/seed${s}/questions.json"
  if [ -f "$q" ]; then
    nq=$(python3 -c "import json;print(len(json.load(open('$q'))['questions']))" 2>/dev/null || echo 0)
    if [ "$nq" = "0" ]; then
      rm -rf "logs/recordings/seed${s}" "data/games_mc/seed${s}_data.json" "data/questions_mc/seed${s}"
      echo "wiped partial world seed$s (0 questions)"
    fi
  fi
}

while true; do
  all_done=1
  for w in C2 D E; do
    range="${RANGES[$w]}"
    incomplete=""
    for s in $range; do
      [ "$(seed_done $s)" = "1" ] || incomplete="$incomplete $s"
    done
    [ -z "$incomplete" ] && continue
    all_done=0
    if ! pgrep -f "stage0_worker.py --seeds ${range// /.*}" > /dev/null && \
       ! pgrep -f "stage0_worker.py --seeds $(echo $range | cut -d' ' -f1)" > /dev/null; then
      # shed E under memory pressure instead of relaunching
      swap_mb=$(sysctl vm.swapusage | grep -oE 'used = [0-9.]+' | grep -oE '[0-9.]+' | cut -d. -f1)
      if [ "$w" = "E" ] && [ "${swap_mb:-0}" -gt 18000 ]; then
        echo "worker E held back (swap ${swap_mb}M)"
        continue
      fi
      for s in $incomplete; do clean_seed $s; done
      nohup env PYTHONPATH=src uv run python scripts/stage0_worker.py \
        --seeds $incomplete --mc-rollouts 40 --max-turns 95 \
        --log tmp/stage0/worker$w.log > /dev/null 2>&1 &
      disown
      echo "relaunched worker $w on:$incomplete"
    fi
  done
  if [ "$all_done" = "1" ]; then
    echo "ALL TEST WORLDS LABELED (348-359 complete)"
    exit 0
  fi
  sleep 300
done
