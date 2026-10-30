#!/usr/bin/env bash
# 6_pull_corpus.sh RUN — step 6: once every worker is gone, count the forks on the archive, pull the run's folder
# (/vol/fbsim_v3/<run>/) to runs/<run>/raw/, check it (corpus_qc.txt) and consolidate it (consol_v4/)
# via finalize.sh, then offer to delete the archive pod. The network volume is kept.
# Next: 7_make_sets.sh RUN --models-csv PATH
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/steps_lib.sh"
step_init "${1:-}"
[ -z "$(live_workers)" ] || die "workers are still live: $(live_workers | tr '\n' ' ')(run 5_finish.sh)"
ARCH="$(grep -v '^#' "$S/archive_pods.txt" 2>/dev/null | grep . | head -n 1)"; [ -n "$ARCH" ] || die "no archive pod in $S/archive_pods.txt"
say "kept anchors: $(kept_seeds)"
say "finalize.sh: archive counts -> pull -> corpus QC -> consolidate (archive pod $ARCH)"
./finalize.sh "$ARCH" 2>&1 | sed 's/^/    /' || die "finalize.sh failed (see above)"
[ -s "$R/consol_v4_SUMMARY.json" ] || die "consolidation produced no $R/consol_v4_SUMMARY.json"
say "per-anchor replays (consol_v4_SUMMARY.json):"
python3 -c "import json,sys; [print(f'    {w}: {v[\"replays\"]} replays, {v[\"truncated\"]} truncated') for w, v in sorted(json.load(open(sys.argv[1])).items())]" "$R/consol_v4_SUMMARY.json"
if confirm "delete the archive pod $ARCH? (the corpus is in $R/raw; the volume and its /vol/fbsim_v3/$FBSIM_RUN/ copy stay)"; then ./rp.sh delete "$ARCH"
else say "archive pod $ARCH left running"; fi
say "next: ./7_make_sets.sh $FBSIM_RUN --models-csv PATH"
