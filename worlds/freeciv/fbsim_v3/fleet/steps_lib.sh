# steps_lib.sh — sourced by the numbered step scripts once HERE is set: `step_init RUN` selects the run (sources
# runs/RUN/shell.sh, then paths.sh) and the helpers below report what is happening.
step_init() {
  local run="${1:-}"
  [ -n "$run" ] || { echo "usage: $(basename "$0") RUN ..." >&2; exit 2; }
  [ -f "$HERE/runs/$run/shell.sh" ] || { echo "no run $run (runs/$run/shell.sh); create it with ./1_new_run.sh" >&2; exit 2; }
  # shellcheck disable=SC1090
  . "$HERE/runs/$run/shell.sh"
  . "$HERE/paths.sh"
  cd "$HERE"
  say "run $FBSIM_RUN: players=$FBSIM_N_PLAYERS fork turn=$FBSIM_FORK_TURN horizons=$FBSIM_HORIZONS seeds=$FBSIM_SEEDS"
}
say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; exit 1; }
confirm() { local a; read -r -p "$1 [y/N] " a; [ "$a" = y ] || [ "$a" = Y ]; }
# live_pod SHORT — ID of this run's live (not deleted) pod named fbsim3-<run>-SHORT, or nothing
live_pod() { [ -f "$STATE/pods.tsv" ] && awk -F'\t' -v n="$POD_PREFIX$1" '$1 !~ /^#/ && $3 == n && $9 != "deleted" {id = $2} END {if (id) print id}' "$STATE/pods.tsv"; }
# live_workers — IDs of this run's live worker pods (fbsim3-<run>-wNN)
live_workers() { [ -f "$STATE/pods.tsv" ] && awk -F'\t' -v p="$POD_PREFIX" '$1 !~ /^#/ && index($3, p "w") == 1 && $9 != "deleted" {print $2}' "$STATE/pods.tsv"; }
# create_pod SHORT FLAVOR VCPU CLOUD [rp.sh create options] — reuse the live pod of that name or create it; prints its ID
create_pod() {
  local short="$1" id out; shift
  id="$(live_pod "$short")"
  if [ -n "$id" ]; then say "pod $POD_PREFIX$short exists: $id" >&2; echo "$id"; return 0; fi
  say "creating pod $POD_PREFIX$short ($1 x $2 vCPU, $3)" >&2
  out="$(./rp.sh create "$short" "$@")" || die "rp.sh create failed: $out"
  say "$out" >&2
  echo "$out" | awk '/^CREATED/ {print $2}'
}
# bringup_once ID — run bringup.sh unless its log already shows a successful bringup
bringup_once() {
  local log="$STATE/bringup_$1.log"
  if grep -q "bringup done" "$log" 2>/dev/null && grep -q "pod_setup_rc=0" "$log"; then say "pod $1 already brought up"; return 0; fi
  say "bringing up pod $1 (waits for ssh, ships the bundle, builds the venv; log: $log)"
  ./bringup.sh "$1" > "$log" 2>&1
  grep -q "pod_setup_rc=0" "$log" && grep -q "bringup done" "$log" || { tail -n 5 "$log" >&2; die "bringup of $1 failed (see $log)"; }
  say "pod $1 is up"
}
# kept_seeds — the seeds in fleet_state/anchors/, sorted
kept_seeds() { ls -d "$STATE"/anchors/seed* 2>/dev/null | sed 's/.*seed//' | sort -n | paste -sd ' ' -; }
