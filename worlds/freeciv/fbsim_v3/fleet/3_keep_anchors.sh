#!/usr/bin/env bash
# 3_keep_anchors.sh RUN SEED... — step 3: keep these anchors. Checks each pulled anchor (data file, fork-turn save,
# settings.txt matching run.env), puts its fork-turn save in fleet_state/anchors/ (the anchor list of every later step;
# seeds not listed are removed from it), logs the choice in fleet_state/ANCHORS.txt, then offers to delete the anchor
# pod. Refused once a queue exists. Next: 4_start_fleet.sh RUN N_WORKERS
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/steps_lib.sh"
step_init "${1:-}"; shift
[ $# -gt 0 ] || die "name the seeds to keep"
[ ! -d "$S/queue" ] || die "the replay queue already exists ($S/queue): the anchors are fixed"
F="$FBSIM_FORK_TURN"
for s in "$@"; do
  d="$S/anchors_raw/seed$s"
  [ -s "$d/seed${s}_data.json.gz" ] || die "seed $s: no pulled data file in $d (run 2_anchors.sh first)"
  ls "$d"/savegames/seed${s}_T${F}_*.sav.xz > /dev/null 2>&1 || die "seed $s: no turn-$F save in $d/savegames"
  grep -qx "players=$FBSIM_N_PLAYERS" "$d/settings.txt" && grep -qx "fork_turn=$F" "$d/settings.txt" \
    || die "seed $s: settings.txt ($(tr '\n' ' ' < "$d/settings.txt")) does not match run.env"
done
mkdir -p "$S/anchors"
for d in "$S"/anchors/seed*; do
  [ -d "$d" ] || continue; s="${d##*seed}"
  case " $* " in *" $s "*) ;; *) say "removing seed $s from the kept set"; rm -rf "$d";; esac
done
for s in "$@"; do
  mkdir -p "$S/anchors/seed$s/savegames"
  rsync -a "$S"/anchors_raw/seed$s/savegames/seed${s}_T${F}_*.sav.xz "$S/anchors/seed$s/savegames/"
done
echo "$(date -u +%FT%TZ) kept $# anchors: $*" >> "$S/ANCHORS.txt"
say "kept $# anchors: $(kept_seeds) (logged in $S/ANCHORS.txt; note the keep rule in runs/$FBSIM_RUN/RUN.md)"
ID="$(live_pod anchors)"
if [ -n "$ID" ]; then
  if confirm "delete the anchor pod $ID? (its anchors are pulled; 4_start_fleet.sh needs it gone)"; then ./rp.sh delete "$ID"
  else say "anchor pod $ID left running"; fi
fi
say "next: ./4_start_fleet.sh $FBSIM_RUN N_WORKERS"
