#!/usr/bin/env bash
# pull.sh ARCHIVE_POD_ID [DEST] — rsync data files + anchor saves + horizon saves (HSAVE_TURNS) of the run's $VOL to local.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/paths.sh"
ID="${1:?archive pod id}"; DEST="${2:-$FBSIM_V3_CORPUS/raw}"
INC=(); for t in $HSAVE_TURNS; do INC+=(--include="*_T${t}_*"); done
read -r IP PORT <<<"$(bash "$HERE/rp.sh" addr "$ID")"; [ -n "$PORT" ] || { echo "no addr"; exit 1; }
mkdir -p "$DEST"
rsync -az --partial --timeout=120 -e "ssh -i '$SSHKEY' -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR" \
  --include='*/' --include='*_data.json.gz' --include='anchors/**' \
  "${INC[@]}" \
  --include='queue/**' --include='logs/**' --include='*.txt' --include='*.rc' --exclude='*' \
  "root@$IP:$VOL/" "$DEST/"
echo "pull rc=$? $(date -u +%FT%TZ) -> $DEST"; du -sh "$DEST"
