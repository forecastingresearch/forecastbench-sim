#!/bin/bash
# finalize_run2.sh PAPER_ROOT [PYTHON] — from the finished run 2 to the paper checkout, in one pass:
#   1. rebuild rows of the batched arm with the current parser (rebuild_rows_v2.py)
#   2. score and tabulate: results/run2_paper/ (build_run2.sh: score_v2, results_table_v2, aggregates)
#   3. deliver into the paper: backing files, FreeCiv tables, figures, FreeCiv cells of the shared tables (--final)
#   4. the run-2 numbers the prose quotes (run2_numbers.py) and the FreeCiv-owned prose and caption edits
#   5. compile the paper with tectonic and print what changed
# Touches only FreeCiv-owned files in the paper.  Nothing is committed or pushed here.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; V="$HERE/.."; PAPER="$1"; PY="${2:-python3}"
BATCHED="$V/results/run2b_2026-09-20/batched"; NATCOND="$V/results/run2_2026-09-20/natcond"
[ -f "$PAPER/main.tex" ] || { echo "no main.tex in $PAPER"; exit 1; }
nc_rows=$(cat "$NATCOND/repinned/results.jsonl" | wc -l | tr -d ' ')
[ "$nc_rows" -ge 1708 ] || { echo "natcond rerun incomplete: $nc_rows of 1708 rows"; exit 1; }
echo "== 1. rows from stored calls"; "$PY" "$HERE/rebuild_rows_v2.py" "$BATCHED" | tail -1
echo "== 2. score and tabulate"; PYTHON="$PY" bash "$HERE/build_run2.sh" run2_paper "$BATCHED" "$NATCOND" | tail -2
echo "== 3. deliver into the paper"; export FBSIM_RUN=run2_paper FBSIM_PAPER_ROOT="$PAPER"
for s in deliver_backing make_freeciv_tables make_freeciv_difficulty_table make_freeciv_family_table make_freeciv_figs; do
  echo "-- $s"; "$PY" "$V/paper/$s.py" 2>&1 | grep -E "^(wrote|copied|removed|edited|unchanged|Traceback|.*Error)" | head -12
done
echo "-- update_shared_tables --final"; "$PY" "$V/paper/update_shared_tables.py" --final 2>&1 | grep -E "^(edited|unchanged|wrote|  [A-Za-z])" | head -14
echo "== 4. numbers and prose"; "$PY" "$HERE/run2_numbers.py" "$BATCHED" "$NATCOND/repinned" "$V/results/run2_paper/run2_numbers.json" | tail -3
"$PY" "$V/paper/apply_run2_prose.py" --paper-root "$PAPER" --numbers "$V/results/run2_paper/run2_numbers.json"
"$PY" "$V/paper/update_captions.py" --paper-root "$PAPER" | head -3
echo "== 5. compile"; ( cd "$PAPER" && tectonic -X compile main.tex > /tmp/tectonic_run2.log 2>&1 && echo "compiled: $(ls -la main.pdf | awk '{print $5}') bytes" || { echo "COMPILE FAILED"; grep -E "^!|error" /tmp/tectonic_run2.log | head -10; } )
( cd "$PAPER" && echo "== changed files" && git status --short | grep -v -E "main\.(pdf|aux|log|bbl|blg|out)$" )
