#!/bin/bash
# Hardened pod creation: platform TTL + printed dead-man bootstrap.
# Usage: bash create_pod.sh <gpu_count> <max_hours> [name] [volume_id]
# Env: RUNPOD_API_KEY (account key, from .env); runpodctl configured.
set -euo pipefail

GPU_COUNT="${1:?gpu count}"
GPU_ID="${GPU_ID:-NVIDIA A40}"  # v2 default: A40 48GB community ($0.35/h); override for H100
HOURS="${2:?max hours}"
NAME="${3:-fbsim-ab}"
VOL="${4:-}"

if date -u -v+1H >/dev/null 2>&1; then
  STOP_AT=$(date -u -v+"${HOURS}"H +%Y-%m-%dT%H:%M:%SZ)
  TERM_AT=$(date -u -v+$((HOURS + 1))H +%Y-%m-%dT%H:%M:%SZ)
else
  STOP_AT=$(date -u -d "+${HOURS} hours" +%Y-%m-%dT%H:%M:%SZ)
  TERM_AT=$(date -u -d "+$((HOURS + 1)) hours" +%Y-%m-%dT%H:%M:%SZ)
fi

ARGS=(pod create --name "$NAME"
  --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
  --gpu-id "$GPU_ID" --gpu-count "$GPU_COUNT"
  --cloud-type SECURE --container-disk-in-gb 40
  --stop-after "$STOP_AT" --terminate-after "$TERM_AT"
  --ports "22/tcp"
  --env "{\"PUBLIC_KEY\":\"$(cat ~/.runpod/ssh/runpodctl-ssh-key.pub | tr -d '\n')\"}")
[ -n "$VOL" ] && ARGS+=(--network-volume-id "$VOL")

echo "Creating: ${GPU_COUNT}x ${GPU_ID}, platform stop at $STOP_AT, terminate at $TERM_AT"
runpodctl "${ARGS[@]}"

cat <<EOF

>>> FIRST COMMAND AFTER SSH (dead-man switch + idle watchdog):
nohup setsid sh -c 'sleep ${HOURS}h; curl -s -X DELETE "https://rest.runpod.io/v1/pods/\$RUNPOD_POD_ID" -H "Authorization: Bearer \$RUNPOD_API_KEY"' >/tmp/deadman.log 2>&1 &
nohup setsid bash -c 'idle=0; while true; do u=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -1); if [ "\$u" -lt 5 ]; then idle=\$((idle+1)); else idle=0; fi; if [ \$idle -ge 20 ]; then curl -s -X DELETE "https://rest.runpod.io/v1/pods/\$RUNPOD_POD_ID" -H "Authorization: Bearer \$RUNPOD_API_KEY"; fi; sleep 60; done' >/tmp/idlewatch.log 2>&1 &
EOF
