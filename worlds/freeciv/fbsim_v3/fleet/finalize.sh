#!/usr/bin/env bash
# finalize.sh ARCHIVE_POD [PULL_POD] — after all workers are gone: archive counts -> pull -> QC -> consolidate_v4.
# Everything goes to the run folder (FBSIM_V3_CORPUS, set by paths.sh). Anchors: FBSIM_ANCHORS (default: the seeds in
# $STATE/anchors/); the player count and turns come from run.env; the ruleset from the pulled anchors. Exits non-zero
# if the pull, the QC or the consolidation fails.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
. "$HERE/paths.sh"
C="$FBSIM_V3_CORPUS"; ARCH="${1:?archive pod id}"; PULLPOD="${2:-$1}"
ANCH="${FBSIM_ANCHORS:-$(ls -d "$STATE"/anchors/seed* 2>/dev/null | sed 's/.*seed//' | sort -n | tr '\n' ' ')}"; [ -n "${ANCH// /}" ] || { echo "no anchors"; exit 1; }
echo "== archive per-anchor counts (forks / sgtables / truncated)"
bash rp.sh ssh "$ARCH" "for a in $ANCH; do echo \"a\$a \$(ls -d $VOL/*/a\$a/forks/rng* 2>/dev/null | wc -l) \$(ls $VOL/*/a\$a/forks/rng*/*_sgtables.json.gz 2>/dev/null | wc -l) \$(ls $VOL/*/a\$a/forks/rng*/TRUNCATED 2>/dev/null | wc -l)\"; done; du -sh $VOL" </dev/null | tee "$C"/archive_counts.txt
echo "== pull (incremental, parallel)"; bash pull2.sh "$PULLPOD" 2>&1 | tail -n 3 || { echo "pull failed"; exit 1; }
echo "== corpus QC"; uv run python ../draw/corpus_qc.py "$C"/raw | tee "$C"/corpus_qc.txt || { echo "corpus QC failed"; exit 1; }
RS="${FBSIM_RULESET:-$(ls "$C"/raw/*/anchors/seed*/ruleset.json 2>/dev/null | head -n 1)}"; [ -s "$RS" ] || { echo "no ruleset.json (set FBSIM_RULESET)"; exit 1; }
echo "== consolidate_v4"; rm -rf "$C"/consol_v4 "$C"/consol_v4_SUMMARY.json; uv run python ../draw/consolidate_v4.py "$C"/raw "$C"/consol_v4 --workers 4 --anchors $ANCH --n-main "$FBSIM_N_PLAYERS" --ruleset "$RS" 2>&1 | tail -n 10 || { echo "consolidation failed"; exit 1; }
echo "== done $(date -u +%FT%TZ)"; ls -la "$C" | head -20
