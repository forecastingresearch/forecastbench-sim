#!/usr/bin/env bash
# anchors_run.sh SEED... — run anchor_one.sh for each seed concurrently (all at once), detached. Player count,
# barbarians and the fork turn (anchors are played to it + 1) come from /workspace/fleet/run.env; seeds outside the
# run's FBSIM_SEEDS range are refused.
[ -f /workspace/fleet/run.env ] || { echo "no /workspace/fleet/run.env (bringup.sh ships it)"; exit 1; }
. /workspace/fleet/run.env
lo="${FBSIM_SEEDS%-*}"; hi="${FBSIM_SEEDS#*-}"
for s in "$@"; do [ "$s" -ge "$lo" ] && [ "$s" -le "$hi" ] || { echo "seed $s outside this run's FBSIM_SEEDS=$FBSIM_SEEDS"; exit 1; }; done
mkdir -p /workspace/fleet/logs /workspace/out/anchors
for s in "$@"; do
  nohup setsid bash /workspace/fleet/anchor_one.sh "$s" > "/workspace/fleet/logs/anchor_seed$s.log" 2>&1 < /dev/null &
  echo "seed$s pid $!"; sleep 2
done
