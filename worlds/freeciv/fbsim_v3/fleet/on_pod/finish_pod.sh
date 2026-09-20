#!/usr/bin/env bash
# finish_pod.sh [WORKERS] — run when this pod's queue is empty and lanes have exited:
# extract exact savegame tables for every published fork, then do a final archive sync and mark FINISHED.
set -u
W="${1:-$(( $(nproc) * 3 / 4 ))}"
F=/workspace/fleet
source /workspace/civbench/pod_env.sh
todo=$(ls $F/queue/todo | wc -l); inprog=$(ls $F/queue/inprog | wc -l)
if [ "$todo" -gt 0 ] || [ "$inprog" -gt 0 ]; then echo "queue not empty: todo=$todo inprog=$inprog"; exit 1; fi
dirs=$(ls -d /workspace/out/a*/forks/rng* 2>/dev/null)
n=$(echo "$dirs" | grep -c .)
echo "$(date -u +%FT%TZ) sg_tables over $n forks with $W workers"
python $F/sg_tables.py $dirs --workers "$W" > $F/logs/sg_tables.log 2>&1
echo "sg_tables rc=$? ok=$(grep -c ' saves, ' $F/logs/sg_tables.log) err=$(grep -vc ' saves, ' $F/logs/sg_tables.log)"
# final sync now (the loop also keeps running)
read -r AIP APORT NAME < <(pgrep -af '/workspace/fleet/sync_loop.sh' | head -1 | awk '{print $(NF-3), $(NF-2), $(NF-1)}')
rsync -a --partial --timeout=300 -e "ssh -i $F/archive_key -p $APORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR" /workspace/out/ "root@$AIP:/vol/fbsim_v3/$NAME/"
echo "final sync rc=$? $(date -u +%FT%TZ)"; date -u +%FT%TZ > $F/FINISHED
