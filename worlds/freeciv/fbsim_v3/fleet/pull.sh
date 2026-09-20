#!/usr/bin/env bash
# pull.sh ARCHIVE_POD_ID [DEST] — rsync data files + anchor saves + horizon saves from the archive volume to local.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ID="${1:?archive pod id}"; DEST="${2:-/Users/jaeholee0404/civbench/tmp/fbsim_v3_corpus/raw}"
read -r IP PORT <<<"$(bash "$HERE/rp.sh" addr "$ID")"; [ -n "$PORT" ] || { echo "no addr"; exit 1; }
mkdir -p "$DEST"
rsync -az --partial --timeout=120 -e "ssh -i $HOME/.ssh/id_ed25519_runpod -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR" \
  --include='*/' --include='*_data.json.gz' --include='anchors/**' \
  --include='*_T60_*' --include='*_T61_*' --include='*_T90_*' --include='*_T120_*' --include='*_T150_*' --include='*_T180_*' --include='*_T210_*' --include='*_T211_*' \
  --include='queue/**' --include='logs/**' --include='*.txt' --include='*.rc' --exclude='*' \
  "root@$IP:/vol/fbsim_v3/" "$DEST/"
echo "pull rc=$? $(date -u +%FT%TZ) -> $DEST"; du -sh "$DEST"
