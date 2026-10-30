#!/usr/bin/env bash
# 7_make_sets.sh RUN --models-csv PATH — step 7: from the consolidated corpus, build the candidate bank (bank_v1/), draw
# the question sets (sets/draw_v1/), attach resolution criteria, write the continuous normalization constants, render
# the fork-turn world reports (reports/) and copy the model panel to run/models_v2.csv (its source is recorded in
# run/models_v2.source.txt). Refuses to redraw once evals/ has results. Next: read the reports, then 8_elicit.sh RUN LABEL
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/steps_lib.sh"
step_init "${1:-}"; shift; CSV=""
while [ $# -gt 0 ]; do case "$1" in --models-csv) CSV="$2"; shift 2;; *) die "bad option $1";; esac; done
[ -n "$CSV" ] || die "pass --models-csv PATH (the model panel, e.g. a models_v2.csv)"
[ -s "$CSV" ] || die "no model panel at $CSV"
[ -d "$R/consol_v4" ] || die "no $R/consol_v4 (run 6_pull_corpus.sh)"
[ -z "$(find "$R/evals" -name results.jsonl 2>/dev/null)" ] || die "$R/evals already has results; redrawing would change the questions they answer"
cd ../draw
step() { say "$1"; shift; uv run python "$@" 2>&1 | grep -vE '^ *(Built|Building|Installed|Uninstalled) ' | sed 's/^/    /'; [ "${PIPESTATUS[0]}" = 0 ] || die "$1 failed"; }
step "candidate bank: every question with its 1,000-replay truth (bank_v1/)" bank_v1.py
step "drawing the question sets (sets/draw_v1/)" draw_v1.py
step "attaching resolution criteria" criteria_v1.py
step "continuous normalization constants" norm_constants_v1.py
step "world reports at turn $FBSIM_FORK_TURN (reports/)" regen_reports.py
mkdir -p "$R/run"; cat "$CSV" > "$R/run/models_v2.csv"
echo "$(cd "$(dirname "$CSV")" && pwd)/$(basename "$CSV") copied $(date -u +%FT%TZ)" > "$R/run/models_v2.source.txt"
say "model panel: $(($(grep -c . "$R/run/models_v2.csv") - 1)) models from $CSV"
say "done. Read the reports in $R/reports/ before eliciting. Next: ./8_elicit.sh $FBSIM_RUN LABEL [--models NAME,...]"
