#!/usr/bin/env bash
# 9_score.sh RUN LABEL — step 9: score every results.jsonl under evals/LABEL/ (including natcond/) against the run's
# sets with score_v2.py into scores/LABEL/ (score_items.csv, score_summary.csv, SCORES.md) and print the headline table.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/steps_lib.sh"
step_init "${1:-}"; shift
LABEL="${1:?usage: 9_score.sh RUN LABEL}"
[ -n "$(find "$R/evals/$LABEL" -name results.jsonl 2>/dev/null)" ] || die "no results under $R/evals/$LABEL (run 8_elicit.sh)"
cd ../run
say "scoring $R/evals/$LABEL into $R/scores/$LABEL"
uv run python score_v2.py "$R/evals/$LABEL" --out "$R/scores/$LABEL" 2>&1 | grep -vE '^ *(Built|Building|Installed|Uninstalled) ' | sed 's/^/    /'
[ "${PIPESTATUS[0]}" = 0 ] || die "score_v2.py failed"
say "headline tables ($R/scores/$LABEL/SCORES.md):"
sed -n '1,40p' "$R/scores/$LABEL/SCORES.md"
