#!/usr/bin/env bash
# pull2.sh ARCHIVE_POD_ID [DEST] [PAR=8] — list wanted files with find ON the archive pod, then pull them with PAR
# parallel rsync --files-from chunks (MooseFS is latency-bound; 8 streams moved 59k files / 2.4 GB in ~5 min).
# Wanted: every *_data.json.gz and *_sgtables.json.gz, TRUNCATED markers, anchors (everything), horizon saves
# (T60/T61/T90/T120/T150/T180/T210/T211) of every fork, queue markers and logs. Incremental (rsync skips existing).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ID="${1:?}"; DEST="${2:-/Users/jaeholee0404/civbench/tmp/fbsim_v3_corpus/raw}"; PAR="${3:-8}"
read -r IP PORT <<<"$(bash "$HERE/rp.sh" addr "$ID")"; [ -n "$PORT" ] || { echo "no addr"; exit 1; }
RSH="ssh -i $HOME/.ssh/id_ed25519_runpod -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR"
mkdir -p "$DEST" "$HERE/state"; L="$HERE/state/pull_files.txt"
echo "$(date -u +%FT%TZ) listing on archive"
$RSH "root@$IP" 'cd /vol/fbsim_v3 && find . \( -path "*/anchors/*" -o -path "*/queue/*" -o -path "*/logs/*" \) -type f -print; find . -maxdepth 5 -type f -not -path "./_excluded/*" \( -name "*_data.json.gz" -o -name "*_sgtables.json.gz" -o -name "TRUNCATED" \) -not -path "*/anchors/*" -print; find . -maxdepth 6 -type f -not -path "./_excluded/*" -path "*/forks/*/savegames/*" \( -name "*_T60_*" -o -name "*_T61_*" -o -name "*_T90_*" -o -name "*_T120_*" -o -name "*_T150_*" -o -name "*_T180_*" -o -name "*_T210_*" -o -name "*_T211_*" \) -print' > "$L"
n=$(wc -l < "$L"); echo "$(date -u +%FT%TZ) $n files listed"
rm -f "$HERE/state/pull_chunk_"*; split -l $(( n / PAR + 1 )) "$L" "$HERE/state/pull_chunk_"
for c in "$HERE"/state/pull_chunk_*; do case "$c" in *.log) continue;; esac
  rsync -a --partial --timeout=300 --files-from="$c" -e "$RSH" "root@$IP:/vol/fbsim_v3/" "$DEST/" > "$c.log" 2>&1 &
done; wait
echo "$(date -u +%FT%TZ) pull done -> $DEST"; du -sh "$DEST"; cat "$HERE"/state/pull_chunk_*.log | grep -v "^$" | head -n 5
