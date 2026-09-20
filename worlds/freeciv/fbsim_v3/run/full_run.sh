#!/usr/bin/env bash
# full_run.sh [MODEL_IDS] — the full v1.8 run: every job for every model (default: all rows of models_v1.csv), one process per
# model into ../fbsim_v3_corpus/elicit_v1/<slug>/ (resumes: rows already there are not re-asked). Workers per model from the
# models file. Watch with: tail -n1 ../fbsim_v3_corpus/elicit_v1/logs/*.log ; health: uv run python smoke_report.py ../fbsim_v3_corpus/elicit_v1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; OUT="$HERE/../fbsim_v3_corpus/elicit_v1"
IDS="${1:-$(tail -n +2 "$HERE/models_v1.csv" | cut -d, -f3 | tr '\n' ',' | sed 's/,$//')}"
bash "$HERE/launch.sh" "$OUT" "$IDS"
