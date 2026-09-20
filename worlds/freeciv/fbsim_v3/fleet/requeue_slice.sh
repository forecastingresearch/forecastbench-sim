#!/usr/bin/env bash
# requeue_slice.sh SRC_POD_ID — deal a dead pod's remaining local todo slice round-robin to live 32-vCPU workers and ship it.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"; SRC="${1:?}"
targets=$(awk -F'\t' -v src="$SRC" '$1!~/^#/ && $9!="deleted" && $5==32 && $2!=src{print $2}' state/pods.tsv)
n=$(echo "$targets" | wc -w); i=0; mkdir -p state/requeue
for t in $targets; do rm -rf "state/requeue/$t"; mkdir -p "state/requeue/$t"; done
for f in state/queue/$SRC/todo/*; do i=$((i+1)); t=$(echo $targets | awk -v k=$(( (i-1) % n + 1 )) '{print $k}'); cp "$f" "state/requeue/$t/"; done
echo "dealt $i tasks from $SRC to $n pods"
for t in $targets; do ( read -r IP PORT <<<"$(bash rp.sh addr $t)"; rsync -az --ignore-existing -e "ssh -i $HOME/.ssh/id_ed25519_runpod -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR" "state/requeue/$t/" "root@$IP:/workspace/fleet/queue/todo/" && cp state/requeue/$t/* state/queue/$t/todo/ && echo "$t +$(ls state/requeue/$t | wc -l)" ) & done; wait
mv state/queue/$SRC state/queue/_requeued_$SRC
