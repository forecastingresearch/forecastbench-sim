#!/usr/bin/env bash
# 4_start_fleet.sh RUN N_WORKERS [--deadman HOURS=48] [--max-retries N=5] — step 4: start the replay fleet. Creates the
# archive pod on the run's network volume (fbsim3-<run>-arch, /vol/fbsim_v3/<run>/) and copies the anchors to it, creates
# N worker pods (fbsim3-<run>-w01..), deals the queue (kept anchors x FBSIM_RNG), brings the workers up in parallel and
# launches their lanes. Workers delete themselves after HOURS whatever their state, so choose it above the expected run
# time. A failed lane launch is retried up to N times per worker.
# Safe to stop and rerun: existing pods, bringups and the queue are reused. Next: 5_finish.sh RUN
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/steps_lib.sh"
step_init "${1:-}"; shift
N="${1:?usage: 4_start_fleet.sh RUN N_WORKERS [--deadman HOURS] [--max-retries N]}"; shift; DEADMAN=48; MAX_RETRIES=5
while [ $# -gt 0 ]; do case "$1" in --deadman) DEADMAN="$2"; shift 2;; --max-retries) MAX_RETRIES="$2"; shift 2;; *) die "bad option $1";; esac; done
[ -z "$(live_pod anchors)" ] || die "the anchor pod $(live_pod anchors) is still live; delete it first (3_keep_anchors.sh asks)"
SEEDS="$(kept_seeds)"; [ -n "$SEEDS" ] || die "no kept anchors in $S/anchors (run 3_keep_anchors.sh)"
lo="${FBSIM_RNG%-*}"; hi="${FBSIM_RNG#*-}"
say "kept anchors: $SEEDS; replays per anchor: $((hi - lo + 1)) (rng $FBSIM_RNG); workers: $N x $FBSIM_POD_VCPU vCPU, deadman ${DEADMAN} h"

# ---- archive pod
ARCH="$(grep -v '^#' "$S/archive_pods.txt" 2>/dev/null | grep . | head -n 1)"
if [ -n "$ARCH" ] && ! awk -F'\t' -v id="$ARCH" '$2 == id && $9 != "deleted" {f = 1} END {exit !f}' "$S/pods.tsv"; then ARCH=""; fi
if [ -z "$ARCH" ]; then
  ARCH="$(create_pod arch "$FBSIM_ARCH_FLAVOR" "$FBSIM_ARCH_VCPU" "$FBSIM_POD_CLOUD" --dc "$FBSIM_VOLUME_DC" --volume "$FBSIM_VOLUME_ID" --deadman 72)"
  [ -n "$ARCH" ] || die "no archive pod"; echo "$ARCH" > "$S/archive_pods.txt"
else say "archive pod: $ARCH"; fi
if [ ! -f "$S/.archive_ready_$ARCH" ]; then
  say "preparing the archive pod $ARCH (archive key, $VOL)"
  ./archive_setup.sh "$ARCH" | sed 's/^/    /'
  say "copying the anchors to $VOL/${POD_PREFIX}anchors/anchors/ (finalize.sh takes the ruleset from there)"
  ./rp.sh ssh "$ARCH" "mkdir -p '$VOL/${POD_PREFIX}anchors/anchors'" || die "ssh to the archive pod failed"
  read -r AIP APORT <<<"$(./rp.sh addr "$ARCH")"
  rsync -az -e "ssh -i '$SSHKEY' -p $APORT ${SSHO[*]}" "$S/anchors_raw/" "root@$AIP:$VOL/${POD_PREFIX}anchors/anchors/" || die "rsync to the archive failed"
  touch "$S/.archive_ready_$ARCH"
fi

# ---- workers
IDS=()
for i in $(seq 1 "$N"); do
  id="$(create_pod "$(printf 'w%02d' "$i")" "$FBSIM_POD_FLAVOR" "$FBSIM_POD_VCPU" "$FBSIM_POD_CLOUD" --deadman "$DEADMAN")"
  [ -n "$id" ] || die "worker $i was not created"; IDS+=("$id")
done
extra="$(live_workers | grep -vxF -f <(printf '%s\n' "${IDS[@]}"))"
[ -z "$extra" ] || say "note: other live workers of this run, not part of this call: $extra"

# ---- queue
if [ -d "$S/queue" ]; then
  have="$(ls "$S/queue" | grep -v '^_' | sort | tr '\n' ' ')"; want="$(printf '%s\n' "${IDS[@]}" | sort | tr '\n' ' ')"
  [ "$have" = "$want" ] || die "the queue in $S/queue is dealt to [$have], not to these workers [$want]; use requeue_slice.sh or rebalance.sh"
  say "queue already dealt ($(find "$S/queue" -type f | wc -l | tr -d ' ') tasks)"
else
  say "dealing the queue"
  uv run python ../draw/make_queue.py --anchors $SEEDS --pods $(printf '%s:'"$FBSIM_POD_VCPU"' ' "${IDS[@]}") | sed 's/^/    /'
fi

# ---- bring up (parallel) and launch
say "bringing up ${#IDS[@]} workers in parallel (logs: $S/bringup_<id>.log)"
for id in "${IDS[@]}"; do ( bringup_once "$id" ) & done; wait
for id in "${IDS[@]}"; do grep -q "pod_setup_rc=0" "$STATE/bringup_$id.log" 2>/dev/null || die "worker $id did not come up; rerun this script after checking its log"; done
say "launching lanes"
while :; do
  ./launch_pending.sh --max-retries "$MAX_RETRIES" | sed 's/^/    /' || die "lane launch gave up (see the ERROR lines above and $S/failed_launch_*.log); rerun with a higher --max-retries"
  n=0; for id in "${IDS[@]}"; do grep -q "lanes alive=" "$S/launch_$id.log" 2>/dev/null && n=$((n + 1)); done
  say "$n of ${#IDS[@]} workers running"
  [ "$n" -eq "${#IDS[@]}" ] && break
  sleep 30
done
./status.sh
say "fleet started. Next: ./5_finish.sh $FBSIM_RUN (waits for the workers to drain, finishes and deletes them)"
