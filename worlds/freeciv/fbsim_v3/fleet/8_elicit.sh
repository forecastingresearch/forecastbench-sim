#!/usr/bin/env bash
# 8_elicit.sh RUN LABEL [--models NAME,...] [other elicit_v2.py options] — step 8: ask the models (OpenRouter) the run's
# questions, batched (elicit_v2.py), into evals/LABEL/; with natural conditionals (FBSIM_NATCOND=1) also the unbatched
# natcond arm (elicit_natcond_v1.py) into evals/LABEL/natcond/. A dry run first prints the call counts and writes
# PREFLIGHT_v2.md; nothing is sent until you confirm. --models defaults to every model in run/models_v2.csv.
# Rerunning with the same LABEL resumes: answered calls are not sent again. Next: 9_score.sh RUN LABEL
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/steps_lib.sh"
step_init "${1:-}"; shift
LABEL="${1:?usage: 8_elicit.sh RUN LABEL [--models NAME,...] [elicit_v2.py options]}"; shift
[[ "$LABEL" =~ ^[A-Za-z0-9_.-]+$ ]] || die "label: letters, digits, _ . - only"
MODELS=""; PASS=()
while [ $# -gt 0 ]; do case "$1" in --models) MODELS="$2"; shift 2;; *) PASS+=("$1"); shift;; esac; done
[ -s "$R/run/models_v2.csv" ] || die "no $R/run/models_v2.csv (run 7_make_sets.sh)"
[ -s "$R/sets/draw_v1/bank_750.json" ] || die "no question sets in $R/sets/draw_v1 (run 7_make_sets.sh)"
[ -n "$MODELS" ] || MODELS="$(python3 -c "import csv,sys; print(','.join(r['name'] for r in csv.DictReader(open(sys.argv[1]))))" "$R/run/models_v2.csv")"
OUT="$R/evals/$LABEL"; NAT=0
[ "$FBSIM_NATCOND" = 1 ] && [ -s "$R/sets/draw_v1/natcond_600.json" ] && NAT=1
cd ../run
run() { uv run python "$@" 2>&1 | grep -vE '^ *(Built|Building|Installed|Uninstalled) ' | sed 's/^/    /'; return "${PIPESTATUS[0]}"; }
say "dry run (no calls): $(echo "$MODELS" | tr ',' '\n' | wc -l | tr -d ' ') models"
run elicit_v2.py --out "$OUT" --models "$MODELS" --dry-run "${PASS[@]}" || die "elicit_v2.py dry run failed"
[ "$NAT" = 1 ] && { run elicit_natcond_v1.py --out "$OUT/natcond" --models "$MODELS" --dry-run || die "elicit_natcond_v1.py dry run failed"; }
say "the prompts are in $OUT/PREFLIGHT_v2.md"
confirm "send these calls to OpenRouter (this costs money)?" || { say "stopped before any call"; exit 0; }
say "eliciting the bank, tails, mirrors and continuous sets into $OUT"
run elicit_v2.py --out "$OUT" --models "$MODELS" "${PASS[@]}" || die "elicit_v2.py failed; rerun to resume"
if [ "$NAT" = 1 ]; then
  say "eliciting the natural conditionals into $OUT/natcond"
  run elicit_natcond_v1.py --out "$OUT/natcond" --models "$MODELS" || die "elicit_natcond_v1.py failed; rerun to resume"
fi
say "done: $(cat "$OUT"/results.jsonl "$OUT"/natcond/results.jsonl 2>/dev/null | wc -l | tr -d ' ') result rows. Next: ./9_score.sh $FBSIM_RUN $LABEL"
