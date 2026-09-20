#!/usr/bin/env bash
# finalize.sh — after all workers are gone: archive counts -> pull -> QC -> consolidate_v4 -> sweep_v4 -> mine_v3.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
C=/Users/jaeholee0404/civbench/tmp/fbsim_v3_corpus; ANCH="7001 7003 7005 7008 7010 7011 7014 7022"
echo "== archive per-anchor counts (forks / sgtables / truncated)"
bash rp.sh ssh 89spdc0ezuuqwi "for a in $ANCH 7016; do echo \"a\$a \$(ls -d /vol/fbsim_v3/*/a\$a/forks/rng* 2>/dev/null | wc -l) \$(ls /vol/fbsim_v3/*/a\$a/forks/rng*/*_sgtables.json.gz 2>/dev/null | wc -l) \$(ls /vol/fbsim_v3/*/a\$a/forks/rng*/TRUNCATED 2>/dev/null | wc -l)\"; done; du -sh /vol/fbsim_v3" </dev/null | tee state/archive_counts.txt
echo "== pull (incremental, parallel)"; bash pull2.sh 3xwo7xrkdsq40e 2>&1 | tail -n 3
echo "== corpus QC"; uv run python corpus_qc.py $C/raw | tee state/corpus_qc.txt
echo "== consolidate_v4"; rm -rf $C/consol_v4; uv run python consolidate_v4.py $C/raw $C/consol_v4 --workers 4 --anchors $ANCH 2>&1 | tail -n 10
echo "== sweep_v4"; uv run python sweep_v4.py $C/consol_v4 $C/sweep_v4.json 2>&1 | tail -n 3
echo "== mine_v3 (legacy families on v4 pickles)"; (cd /Users/jaeholee0404/civbench/tmp/fbsim_v2/mining && uv run python mine_v3.py $C/consol_v4 $C/mined_v4_legacy.json 2>&1 | tail -n 3)
cp CHANNEL_READINESS.md state/RUN_LOG.md state/archive_counts.txt state/corpus_qc.txt $C/ 2>/dev/null
echo "== done $(date -u +%FT%TZ)"; ls -la $C | head -20
