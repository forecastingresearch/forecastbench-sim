#!/usr/bin/env bash
# 2_anchors.sh RUN [SEED ...] — step 2: play the run's anchor games on one pod (fbsim3-<run>-anchors), as many at a time
# as the pod has civservers, then pull them to fleet_state/anchors_raw/ and print the anchor QC table.
# Seeds default to the run's FBSIM_SEEDS range. Waits until every anchor has finished; safe to stop and rerun (the pod,
# its bringup and finished anchors are reused). Next: 3_keep_anchors.sh RUN SEED...
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/steps_lib.sh"
step_init "${1:-}"; shift
if [ $# -gt 0 ]; then SEEDS=("$@"); else SEEDS=($(seq "${FBSIM_SEEDS%-*}" "${FBSIM_SEEDS#*-}")); fi
for s in "${SEEDS[@]}"; do [ "$s" -ge "${FBSIM_SEEDS%-*}" ] && [ "$s" -le "${FBSIM_SEEDS#*-}" ] || die "seed $s is outside the run's FBSIM_SEEDS=$FBSIM_SEEDS"; done
say "anchors to play: ${#SEEDS[@]} (seeds ${SEEDS[0]}..${SEEDS[${#SEEDS[@]}-1]}), each to turn $((FBSIM_FORK_TURN + 1))"

ID="$(create_pod anchors "$FBSIM_POD_FLAVOR" "$FBSIM_POD_VCPU" "$FBSIM_POD_CLOUD")"; [ -n "$ID" ] || die "no anchor pod"
bringup_once "$ID"
CIV="$(sed -n 's/^civservers=//p' "$STATE/bringup_$ID.log" | tail -n 1)"
[ "${CIV:-0}" -gt 0 ] || die "the bringup log shows no civservers"
say "the pod runs $CIV civservers: at most $CIV anchors at a time"

while :; do
  st="$(./rp.sh ssh "$ID" bash -s -- "${SEEDS[@]}" < on_pod/anchor_status.sh 2>/dev/null)"
  if [ -z "$st" ]; then say "pod $ID did not answer; retrying in 60 s"; sleep 60; continue; fi
  pending=($(awk '$2 == "pending" {print $1}' <<<"$st")); running=$(grep -c ' running$' <<<"$st")
  ndone=$(grep -c ' done$' <<<"$st"); nfail=$(grep -c ' failed$' <<<"$st")
  say "anchors: $ndone done, $running running, $nfail failed, ${#pending[@]} waiting"
  if [ ${#pending[@]} -gt 0 ] && [ "$running" -lt "$CIV" ]; then
    start=("${pending[@]:0:$((CIV - running))}")
    say "starting ${start[*]}"
    ./rp.sh ssh "$ID" /workspace/fleet/anchors_run.sh "${start[@]}" > /dev/null || say "starting failed; will retry"
  elif [ ${#pending[@]} -eq 0 ] && [ "$running" -eq 0 ]; then break
  fi
  sleep 60
done
failed=($(awk '$2 == "failed" {print $1}' <<<"$st"))
[ ${#failed[@]} -eq 0 ] || say "failed anchors (log: /workspace/fleet/logs/anchor_seed<N>.log on $ID): ${failed[*]}"

say "pulling the anchors to $S/anchors_raw/"
read -r IP PORT <<<"$(./rp.sh addr "$ID")"; [ -n "${PORT:-}" ] || die "no address for $ID"
mkdir -p "$S/anchors_raw"
rsync -az -e "ssh -i '$SSHKEY' -p $PORT ${SSHO[*]}" "root@$IP:/workspace/out/anchors/" "$S/anchors_raw/" || die "rsync failed"
ls "$S"/anchors_raw/seed*/seed*_data.json.gz > /dev/null 2>&1 || die "no anchor was published; see the logs on the pod"
say "anchor QC (also in $S/anchor_qc.txt)"
uv run python ../draw/anchor_qc.py "$S"/anchors_raw/seed*/seed*_data.json.gz | tee "$S/anchor_qc.txt"
say "done. Choose the anchors to keep, write the rule down, then: ./3_keep_anchors.sh $FBSIM_RUN SEED..."
