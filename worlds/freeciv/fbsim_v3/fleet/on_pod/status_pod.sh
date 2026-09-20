#!/usr/bin/env bash
# printed as one line of key=value pairs
F=/workspace/fleet; Q=$F/queue
alive=0; if [ -f $F/lanes.pids ]; then while read -r n p; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done < $F/lanes.pids; fi
todo=$(ls $Q/todo 2>/dev/null | wc -l); inprog=$(ls $Q/inprog 2>/dev/null | wc -l); done_=$(ls $Q/done 2>/dev/null | wc -l); failed=$(ls $Q/failed 2>/dev/null | wc -l)
recent=$(find $Q/done -type f -mmin -30 2>/dev/null | wc -l)
load=$(cut -d' ' -f1 /proc/loadavg); free=$(df -m /workspace | awk 'NR==2{print $4}')
sync=$(tail -1 $F/logs/sync.log 2>/dev/null | awk '{print $NF}')
forks=$(pgrep -c -f 'run_fork[.]py')
echo "lanes=$alive todo=$todo inprog=$inprog done=$done_ failed=$failed done30m=$recent forks_running=$forks load=$load free_mb=$free sync=${sync:-none} nproc=$(nproc)"
