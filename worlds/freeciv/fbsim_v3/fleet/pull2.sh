#!/usr/bin/env bash
# pull2.sh ARCHIVE_POD_ID [DEST] [PAR=8] — list wanted files with find ON the archive pod, then pull them with PAR
# parallel rsync --files-from chunks (MooseFS is latency-bound; 8 streams moved 59k files / 2.4 GB in ~5 min).
# Wanted, under the run's $VOL: every *_data.json.gz and *_sgtables.json.gz, TRUNCATED markers, anchors (everything),
# the horizon saves of every fork (HSAVE_TURNS: T60/T61/T90/.../T210/T211 in the paper), queue markers and logs.
# Incremental (rsync skips existing).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ID="${1:?}"; . "$HERE/paths.sh"
DEST="${2:-$FBSIM_V3_CORPUS/raw}"; PAR="${3:-8}"
HS=""; for t in $HSAVE_TURNS; do HS="$HS${HS:+ -o} -name \"*_T${t}_*\""; done
read -r IP PORT <<<"$(bash "$HERE/rp.sh" addr "$ID")"; [ -n "$PORT" ] || { echo "no addr"; exit 1; }
RSH="ssh -i '$SSHKEY' -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR"
mkdir -p "$DEST" "$STATE"; L="$STATE/pull_files.txt"
echo "$(date -u +%FT%TZ) listing on archive"
podsh "$PORT" "root@$IP" "cd '$VOL' && "'find . \( -path "*/anchors/*" -o -path "*/queue/*" -o -path "*/logs/*" \) -type f -print; find . -maxdepth 5 -type f -not -path "./_excluded/*" \( -name "*_data.json.gz" -o -name "*_sgtables.json.gz" -o -name "TRUNCATED" \) -not -path "*/anchors/*" -print; find . -maxdepth 6 -type f -not -path "./_excluded/*" -path "*/forks/*/savegames/*" \( '"$HS"' \) -print' > "$L" || { echo "listing on the archive failed"; exit 1; }
n=$(wc -l < "$L"); echo "$(date -u +%FT%TZ) $n files listed"; [ "$n" -gt 0 ] || { echo "nothing to pull"; exit 1; }
rm -f "$STATE/pull_chunk_"*; pids=(); split -l $(( n / PAR + 1 )) "$L" "$STATE/pull_chunk_"
for c in "$STATE"/pull_chunk_*; do case "$c" in *.log) continue;; esac
  rsync -a --partial --timeout=300 --files-from="$c" -e "$RSH" "root@$IP:$VOL/" "$DEST/" > "$c.log" 2>&1 & pids+=($!)
done; fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "$(date -u +%FT%TZ) pull done -> $DEST"; du -sh "$DEST"; cat "$STATE"/pull_chunk_*.log | grep -v "^$" | head -n 5
[ "$fail" = 0 ] || { echo "a pull chunk failed (see $STATE/pull_chunk_*.log)"; exit 1; }
