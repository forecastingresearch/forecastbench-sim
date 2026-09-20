#!/bin/bash
# build_run2.sh RUN_DIR — score run 2 and build the paper's result files from it.
#   RUN_DIR (under ../results/) holds batched/<model>/results.jsonl (elicit_v2.py, arms t1) and
#   natcond/from_run1/results.jsonl + natcond/repinned/results.jsonl (arms t1nc, t2, nonews).
#   Writes RUN_DIR/scores_v1/ (score_v2.py: SCORES.md, score_items.csv.gz, score_summary.csv) and
#   RUN_DIR/results_v1/ (results_table_v2.py: wide, long, binary, continuous CSVs and the markdown table),
#   the folder names the paper generators expect (worlds/freeciv/fbsim_v3/paper/_common.py).
#   bash build_run2.sh run2_2026-09-20        then: FBSIM_RUN=run2_2026-09-20 python ../paper/<generator>.py
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; RUN="$HERE/../results/$1"; PY="${PYTHON:-python3}"
[ -d "$RUN/batched" ] || { echo "no $RUN/batched"; exit 1; }
"$PY" "$HERE/score_v2.py" "$RUN/batched" "$RUN/natcond" --out "$RUN/scores_v1" --boot 1000 --seed 0
gzip -9 -f "$RUN/scores_v1/score_items.csv"
"$PY" "$HERE/results_table_v2.py" "$RUN/batched" "$RUN/natcond" --out "$RUN/results_v1"
cp "$HERE/../results/run1_2026-09-09/model_scores.csv" "$HERE/../results/run1_2026-09-09/model_scores_with_slugs.csv" "$RUN/"
echo "scored and tabulated $RUN; next: FBSIM_RUN=$1 FBSIM_PAPER_ROOT=<paper> python ../paper/{deliver_backing,make_freeciv_tables,make_freeciv_difficulty_table,make_freeciv_family_table,make_freeciv_figs,update_shared_tables}.py"
echo "note: family_horizon_scores.csv and reliability_bands.csv for the figures come from score_items (make_freeciv_aggregates.py, to be written for run 2)"
