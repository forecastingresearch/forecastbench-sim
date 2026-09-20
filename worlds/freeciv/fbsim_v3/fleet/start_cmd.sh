set -x; umask 077
printenv PUBLIC_KEY > /tmp/f3_pubkey; printenv RUNPOD_POD_ID > /tmp/f3_podid; printenv RUNPOD_API_KEY > /tmp/f3_apikey; printenv DEADMAN_HOURS > /tmp/f3_deadman
cat > /tmp/f3_root.sh <<'ROOT'
set -x; mkdir -p /workspace/fleet/logs; chmod 600 /tmp/f3_apikey
cat > /workspace/fleet/deadman.sh <<'EOD'
#!/bin/bash
sleep "$(cat /tmp/f3_deadman)h"
curl -fsS -X DELETE "https://rest.runpod.io/v1/pods/$(cat /tmp/f3_podid)" -H "Authorization: Bearer $(cat /tmp/f3_apikey)"
EOD
chmod 700 /workspace/fleet/deadman.sh
nohup setsid /workspace/fleet/deadman.sh > /workspace/fleet/logs/deadman.log 2>&1 < /dev/null &
sleep 1; pgrep -f 'fleet/deadman[.]sh' > /workspace/fleet/DEADMAN_ARMED || true
export DEBIAN_FRONTEND=noninteractive; apt-get update -qq; apt-get install -yqq openssh-server curl ca-certificates rsync procps
mkdir -p /root/.ssh /run/sshd; cat /tmp/f3_pubkey > /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys
/usr/sbin/sshd; touch /ready
ROOT
if [ "$(id -u)" = 0 ]; then bash /tmp/f3_root.sh; elif sudo -n true 2>/dev/null; then sudo -n bash /tmp/f3_root.sh; else echo docker | sudo -S -p '' bash /tmp/f3_root.sh; fi
sleep infinity
