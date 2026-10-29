#!/usr/bin/env bash
# 1_new_run.sh RUN --seeds LO-HI [--players 5] [--barbarians DISABLED] [--fork-turn 60] [--horizons H,...]
#   [--replays N | --rng 5001-6000] [--natcond 0|1] [--flavor cpu3c] [--vcpu 32] [--cloud SECURE] [--arch-flavor cpu3c] [--arch-vcpu 8]
#   [--volume ID] [--volume-dc DC] [--force]
# Step 1: create data/freeciv/fbsim_v3/runs/RUN/ with its run.env, an empty fleet_state/ and shell.sh, which a shell
# sources to export FBSIM_RUN and set the runbook's shorthands (R, S, K, V3, REPO). --flavor/--vcpu/--cloud are the
# anchor and worker pods, --arch-* the archive pod, --volume/--volume-dc its network volume (default: volume.txt).
# Replays per anchor = the rng range's size: --replays N is --rng 5001-<5000+N> (default 1,000).
# Horizons default to the paper's offsets from the fork turn (+30 ... +150); natcond defaults to 1 with two or more
# players. Refuses an existing run, anchor seeds that overlap another run's (world ids are seed<N>), and a checkout
# with uncommitted or untracked changes unless --force (run.env records the commit, and pods get the working tree).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../../../.." && pwd)"
RUNS="$REPO/data/freeciv/fbsim_v3/runs"
RUN="${1:?usage: 1_new_run.sh RUN --seeds LO-HI [options]}"; shift
[[ "$RUN" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "run name: letters, digits, _ and - only" >&2; exit 2; }
NP=5 BARB="" FORK=60 HZ="" RNG=5001-6000 NC="" SEEDS="" FORCE=0
FLAVOR=cpu3c VCPU=32 CLOUD=SECURE AFLAVOR=cpu3c AVCPU=8 VOLID="" VOLDC=""
VT="$REPO/data/freeciv/fbsim_v3/volume.txt"
[ -f "$VT" ] && { VOLID="$(sed -n 's/^ID=//p' "$VT")"; VOLDC="$(sed -n 's/^DC=//p' "$VT")"; }
while [ $# -gt 0 ]; do case "$1" in
  --force) FORCE=1; shift; continue;;
  --players) NP="$2";; --barbarians) BARB="$2";; --fork-turn) FORK="$2";; --horizons) HZ="$2";;
  --rng) RNG="$2";; --replays) RNG="5001-$((5000 + $2))";; --natcond) NC="$2";; --seeds) SEEDS="$2";;
  --flavor) FLAVOR="$2";; --vcpu) VCPU="$2";; --cloud) CLOUD="$2";; --arch-flavor) AFLAVOR="$2";; --arch-vcpu) AVCPU="$2";;
  --volume) VOLID="$2";; --volume-dc) VOLDC="$2";; *) echo "bad option $1" >&2; exit 2;; esac; shift 2; done
[ -n "$VOLID" ] && [ -n "$VOLDC" ] || { echo "no network volume: pass --volume ID --volume-dc DC (or fill $VT)" >&2; exit 2; }
[[ "$SEEDS" =~ ^[0-9]+-[0-9]+$ ]] || { echo "--seeds LO-HI is required (the anchor seeds this run may use)" >&2; exit 2; }
[[ "$RNG" =~ ^[0-9]+-[0-9]+$ ]] && [ "${RNG%-*}" -le "${RNG#*-}" ] || { echo "bad rng range $RNG (--rng LO-HI or --replays N)" >&2; exit 2; }
[ -n "$HZ" ] || HZ="$((FORK + 30)),$((FORK + 60)),$((FORK + 90)),$((FORK + 120)),$((FORK + 150))"
[ -n "$NC" ] || { [ "$NP" -gt 1 ] && NC=1 || NC=0; }
[ "${HZ%%,*}" -gt "$FORK" ] || { echo "horizons must come after the fork turn" >&2; exit 2; }
D="$RUNS/$RUN"; [ ! -e "$D" ] || { echo "$D exists" >&2; exit 1; }
DIRTY="$(git -C "$REPO" status --porcelain)"
if [ -n "$DIRTY" ] && [ "$FORCE" = 0 ]; then
  { echo "the checkout has uncommitted or untracked changes; commit them first or pass --force:"; echo "$DIRTY" | head -n 20; } >&2; exit 1
fi
COMMIT="$(git -C "$REPO" rev-parse --short HEAD)${DIRTY:+ with uncommitted changes (--force)}"
lo="${SEEDS%-*}"; hi="${SEEDS#*-}"
for f in "$RUNS"/*/run.env; do
  [ -f "$f" ] || continue
  s="$(sed -n 's/^FBSIM_SEEDS=//p' "$f")"; [ -n "$s" ] || continue
  if [ "$lo" -le "${s#*-}" ] && [ "${s%-*}" -le "$hi" ]; then echo "seeds $SEEDS overlap $s of $(basename "$(dirname "$f")")" >&2; exit 1; fi
done
mkdir -p "$D/fleet_state"
cat > "$D/run.env" <<EOF
# FreeCiv v3 run $RUN, created $(date -u +%FT%TZ) at commit $COMMIT. Read by
# fleet/paths.sh, the pods (bringup.sh ships it) and runcfg.py. Change it only before the first anchor is played.
FBSIM_RUN=$RUN
FBSIM_N_PLAYERS=$NP
FBSIM_BARBARIANS=$BARB
FBSIM_FORK_TURN=$FORK
FBSIM_HORIZONS=$HZ
FBSIM_RNG=$RNG
FBSIM_NATCOND=$NC
FBSIM_SEEDS=$SEEDS
FBSIM_POD_FLAVOR=$FLAVOR
FBSIM_POD_VCPU=$VCPU
FBSIM_POD_CLOUD=$CLOUD
FBSIM_ARCH_FLAVOR=$AFLAVOR
FBSIM_ARCH_VCPU=$AVCPU
FBSIM_VOLUME_ID=$VOLID
FBSIM_VOLUME_DC=$VOLDC
EOF
cat > "$D/shell.sh" <<EOF
# source this file to work on run $RUN: FBSIM_RUN for the fleet and Python scripts, and the runbook's shorthands.
export FBSIM_RUN=$RUN
REPO="$REPO"
V3="\$REPO/worlds/freeciv/fbsim_v3"
R="$D"
S="\$R/fleet_state"
K="\$REPO/credentials/freeciv"
EOF
echo "created $D"; cat "$D/run.env"
echo
echo "next: ./2_anchors.sh $RUN   (for manual commands: source runs/$RUN/shell.sh)"
