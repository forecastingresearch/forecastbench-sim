#!/usr/bin/env bash
# archive_setup.sh POD_ID — prepare the archive pod: accept the fleet's archive key, create /vol/fbsim_v3, report volume state.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ID="${1:?}"
for i in $(seq 1 40); do bash "$HERE/rp.sh" ssh "$ID" true 2>/dev/null && break; sleep 15; done
PUB="$(cat "$HERE/state/archive_key.pub")"
bash "$HERE/rp.sh" ssh "$ID" "grep -qF '$PUB' /root/.ssh/authorized_keys || echo '$PUB' >> /root/.ssh/authorized_keys; mkdir -p /vol/fbsim_v3; df -h /vol | tail -1; echo '--- volume top level:'; ls -la /vol | head -20; du -sh /vol/* 2>/dev/null | sort -h | tail -12; pgrep -af 'fleet/deadman[.]sh' | head -1"
