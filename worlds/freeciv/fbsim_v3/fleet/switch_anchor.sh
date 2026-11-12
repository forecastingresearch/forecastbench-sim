#!/usr/bin/env bash
# switch_anchor.sh OLD NEW — drop OLD's remaining todo tasks on every worker, place NEW's fork-turn save, deal NEW's forks (the run's FBSIM_RNG).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"; OLD="$1"; NEW="$2"; 
. "$HERE/paths.sh"
[ -s "$STATE"/archive_pods.txt ] || { echo "no archive pods in $STATE/archive_pods.txt"; exit 1; }
mkdir -p "$STATE/anchors/seed$NEW/savegames"; cp "$STATE"/anchors_raw/seed$NEW/savegames/seed${NEW}_T${FBSIM_FORK_TURN}_*.sav.xz "$STATE/anchors/seed$NEW/savegames/"
pods=$(awk -F'\t' 'NR==FNR{arch[$1]=1; next} $1!~/^#/ && $9!="deleted" && !($2 in arch){printf "%s:%s ", $2, $5}' "$STATE"/archive_pods.txt "$STATE"/pods.tsv)
python3 ../draw/make_queue.py --anchors "$NEW" --rng "$FBSIM_RNG" --pods $pods --out "$STATE/queue_$NEW" | tail -n 1
for spec in $pods; do id="${spec%%:*}"
  ( read -r IP PORT <<<"$(bash rp.sh addr "$id")"
    RSH="ssh -i '$SSHKEY' -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR"
    podsh "$PORT" "root@$IP" "mkdir -p /workspace/civbench/data/freeciv/logs/recordings/seed$NEW/savegames /workspace/fleet/queue/dropped"
    rsync -az -e "$RSH" "$STATE/anchors/seed$NEW/savegames/" "root@$IP:/workspace/civbench/data/freeciv/logs/recordings/seed$NEW/savegames/"
    rsync -az --ignore-existing -e "$RSH" "$STATE/queue_$NEW/$id/todo/" "root@$IP:/workspace/fleet/queue/todo/"
    out="$(podsh "$PORT" "root@$IP" "chmod -R a+rX /workspace/civbench/data/freeciv/logs/recordings/seed$NEW; n=0; for f in /workspace/fleet/queue/todo/a$OLD-*; do [ -e \"\$f\" ] || continue; mv \"\$f\" /workspace/fleet/queue/dropped/ && n=\$((n+1)); done; echo dropped_$OLD=\$n added_$NEW=\$(ls /workspace/fleet/queue/todo | grep -c \"^a$NEW-\") anchor_$NEW=\$(ls /workspace/civbench/data/freeciv/logs/recordings/seed$NEW/savegames | wc -l)")"
    echo "$id $out" ) &
done; wait
echo "$(date -u +%FT%TZ) switched anchor $OLD -> $NEW (remaining $OLD todo dropped; finished $OLD forks kept on archive as extra)" >> "$STATE"/ANCHORS.txt
