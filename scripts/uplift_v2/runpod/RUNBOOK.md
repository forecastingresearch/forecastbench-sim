# Hardened RunPod launch runbook (verified live 2026-07-13)

Post-incident cost controls. Key verified facts:

- **No account spend cap exists** (only an $80/hr fraud limit). Enable the console
  Billing → "Low balance alert"; keep auto-pay OFF; keep prepaid balance = run cost
  + ~30% so RunPod's force-stop-at-$0 is the final backstop.
- **Platform TTL at creation**: `runpodctl pod create --stop-after / --terminate-after`
  take **RFC3339 timestamps** (NOT "4h" — docs claiming duration syntax are wrong;
  verified against the GraphQL schema). Validate once on a cheap pod.
- **In-pod dead-man switch**: `poweroff` does NOT work (containers), and an exited
  container still bills. Every pod has `runpodctl` + pod-scoped `$RUNPOD_API_KEY` +
  `$RUNPOD_POD_ID` preinstalled. Use:
  `nohup setsid sh -c 'sleep 4h; curl -s -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY"' >/tmp/deadman.log 2>&1 &`
- **Idle watchdog** (no built-in for pods): loop nvidia-smi; terminate after N idle
  minutes (see fw of scripts/uplift_v2/runpod/create_pod.sh).
- **Stop vs terminate**: stopped pods bill volume disk at 2× ($0.20/GB/mo) and may
  lose GPU reservation; with a **network volume** ($0.07/GB/mo, Secure Cloud only),
  /workspace survives TERMINATE — prefer terminate + network volume.
- **Cheap bring-up**: create network volume in a DC stocking 8×H100 SXM → 1×H100
  pod ($2.99/hr) for env setup → terminate → 8× pod ($23.92/hr secure, $21.52
  community-but-no-network-volume) only for the verified run.
- **Spot**: currently zero discount on H100 SXM (spot == on-demand price) — use
  on-demand.

## Launch procedure

1. Low-balance alert on; balance minimal.
2. `bash scripts/uplift_v2/runpod/create_pod.sh <gpu_count> <hours>` — sets
   platform TTL and prints the pod id; first SSH command installs dead-man switch
   + idle watchdog (the script prints it).
3. External check every ~30 min (heartbeat or cron):
   `curl -s https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RUNPOD_API_KEY"`
   → DELETE anything past deadline.

Five layers: platform TTL → in-pod timer → idle watchdog → external checks →
small prepaid balance. Any single one caps a runaway at ~1 hour.
