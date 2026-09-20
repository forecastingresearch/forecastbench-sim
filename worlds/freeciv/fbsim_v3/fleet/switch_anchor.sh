#!/usr/bin/env bash
# switch_anchor.sh OLD NEW — drop OLD's remaining todo tasks on every worker, place NEW's T60 save, deal NEW's 1000 forks.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"; OLD="$1"; NEW="$2"; SSHKEY="$HOME/.ssh/id_ed25519_runpod"
mkdir -p "state/anchors/seed$NEW/savegames"; cp state/anchors_raw/seed$NEW/savegames/seed${NEW}_T60_*.sav.xz "state/anchors/seed$NEW/savegames/"
pods=$(awk -F'\t' '$1!~/^#/ && $9!="deleted" && $2!="3xwo7xrkdsq40e" && $2!="89spdc0ezuuqwi" && $2!="mks2924lqlbh6a"{printf "%s:%s ", $2, $5}' state/pods.tsv)
python3 make_queue.py --anchors "$NEW" --rng 5001-6000 --pods $pods --out "state/queue_$NEW" | tail -n 1
for spec in $pods; do id="${spec%%:*}"
  ( read -r IP PORT <<<"$(bash rp.sh addr "$id")"
    RSH="ssh -i $SSHKEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR"
    $RSH "root@$IP" "mkdir -p /workspace/civbench/worlds/freeciv/logs/recordings/seed$NEW/savegames /workspace/fleet/queue/dropped"
    rsync -az -e "$RSH" "state/anchors/seed$NEW/savegames/" "root@$IP:/workspace/civbench/worlds/freeciv/logs/recordings/seed$NEW/savegames/"
    rsync -az --ignore-existing -e "$RSH" "state/queue_$NEW/$id/todo/" "root@$IP:/workspace/fleet/queue/todo/"
    out="$($RSH "root@$IP" "chmod -R a+rX /workspace/civbench/worlds/freeciv/logs/recordings/seed$NEW; n=0; for f in /workspace/fleet/queue/todo/a$OLD-*; do [ -e \"\$f\" ] || continue; mv \"\$f\" /workspace/fleet/queue/dropped/ && n=\$((n+1)); done; echo dropped_$OLD=\$n added_$NEW=\$(ls /workspace/fleet/queue/todo | grep -c \"^a$NEW-\") anchor_$NEW=\$(ls /workspace/civbench/worlds/freeciv/logs/recordings/seed$NEW/savegames | wc -l)")"
    echo "$id $out" ) &
done; wait
echo "$(date -u +%FT%TZ) switched anchor $OLD -> $NEW (remaining $OLD todo dropped; finished $OLD forks kept on archive as extra)" >> state/ANCHORS.txt
