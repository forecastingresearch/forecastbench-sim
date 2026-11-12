#!/usr/bin/env bash
# 5_finish.sh RUN [--interval SECONDS=300] — step 5: every interval, print the fleet status and run finish_all.sh, which
# finishes each worker as soon as its queue is empty (exact savegame tables, final archive sync) and deletes it once the
# archive holds all its forks. Lanes try each task three times (the last with a 90-minute timeout); the tasks that
# failed every time are listed at the end. Safe to stop and rerun. Next: 6_pull_corpus.sh RUN
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/steps_lib.sh"
step_init "${1:-}"; shift; IV=300
while [ $# -gt 0 ]; do case "$1" in --interval) IV="$2"; shift 2;; *) die "bad option $1";; esac; done
[ -n "$(live_workers)" ] || { say "no live workers; next: ./6_pull_corpus.sh $FBSIM_RUN"; exit 0; }
while :; do
  ./status.sh | sed 's/^/    /'
  ./finish_all.sh | grep -E "START|DELETE|UNREACHABLE|kept" | sed 's/^/    /'
  [ -n "$(live_workers)" ] || break
  say "$(live_workers | wc -l | tr -d ' ') workers left; next check in $IV s"; sleep "$IV"
done
n=$(sort -u "$S/failed_tasks.txt" 2>/dev/null | grep -c .)
say "tasks that failed every attempt: ${n:-0}$( [ "${n:-0}" -gt 0 ] && echo " ($S/failed_tasks.txt; each leaves its anchor one replay short)")"
[ "${n:-0}" -gt 0 ] && sort -u "$S/failed_tasks.txt" | tr '\n' ' ' | sed 's/^/    /' && echo
say "all workers are finished and deleted. Next: ./6_pull_corpus.sh $FBSIM_RUN"
