#!/usr/bin/env bash
# rp.sh — minimal RunPod REST helper for the hand-driven fbsim v3 rerun (2026-09-02).
#   rp.sh create NAME FLAVOR VCPU CLOUD [--dc DC] [--volume ID] [--deadman H] [--disk GB] [--start FILE]
#   rp.sh list | get ID | addr ID | ssh ID [CMD...] | delete ID | ledger
# Ledger: state/pods.tsv (created_at, id, name, flavor, vcpu, cloud, dc, cost, status)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$HERE/state"; mkdir -p "$STATE"
LEDGER="$STATE/pods.tsv"
API="https://rest.runpod.io/v1"
KEY="${RUNPOD_API_KEY:-$(grep -E '^RUNPOD_API_KEY=' /Users/jaeholee0404/civbench/.env | head -1 | cut -d= -f2- | tr -d '"'"'"' \r')}"
PUBKEY="$(cat "$HOME/.ssh/id_ed25519_runpod.pub")"
SSHKEY="$HOME/.ssh/id_ed25519_runpod"
IMAGE="civrealm/freeciv-web:1.4"
[ -n "$KEY" ] || { echo "no RUNPOD_API_KEY" >&2; exit 1; }

HTTP=""
rp() { # rp METHOD PATH [JSON]  -> body on stdout, HTTP set
  local m="$1" p="$2" body="${3:-}" out
  if [ -n "$body" ]; then
    out="$(curl -sS -m 120 -X "$m" "$API$p" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d "$body" -w $'\n%{http_code}' || echo $'\n000')"
  else
    out="$(curl -sS -m 60 -X "$m" "$API$p" -H "Authorization: Bearer $KEY" -w $'\n%{http_code}' || echo $'\n000')"
  fi
  HTTP="$(printf '%s' "$out" | tail -n1)"; echo "$HTTP" > "$STATE/.http"; printf '%s\n' "$out" | sed '$d'
}
http() { cat "$STATE/.http" 2>/dev/null; }

cmd_create() {
  local NAME="$1" FLAVOR="$2" VCPU="$3" CLOUD="$4"; shift 4
  local DC="" VOL="" DEADMAN=18 DISK=40 STARTF="$HERE/start_cmd.sh"
  while [ $# -gt 0 ]; do case "$1" in
    --dc) DC="$2"; shift 2;; --volume) VOL="$2"; shift 2;; --deadman) DEADMAN="$2"; shift 2;;
    --disk) DISK="$2"; shift 2;; --start) STARTF="$2"; shift 2;; *) echo "bad opt $1" >&2; exit 2;; esac; done
  local SCRIPT; SCRIPT="$(cat "$STARTF")"
  local payload
  payload="$(NAME="$NAME" IMAGE="$IMAGE" FLAVOR="$FLAVOR" VCPU="$VCPU" CLOUD="$CLOUD" DC="$DC" VOL="$VOL" \
             DEADMAN="$DEADMAN" DISK="$DISK" PUBKEY="$PUBKEY" SCRIPT="$SCRIPT" python3 - <<'PY'
import json, os
e = os.environ
d = {"name": e["NAME"], "imageName": e["IMAGE"], "computeType": "CPU",
     "cpuFlavorIds": [e["FLAVOR"]], "vcpuCount": int(e["VCPU"]), "cloudType": e["CLOUD"],
     "containerDiskInGb": int(e["DISK"]), "ports": ["22/tcp"],
     "env": {"PUBLIC_KEY": e["PUBKEY"], "DEADMAN_HOURS": e["DEADMAN"]},
     "dockerStartCmd": ["bash", "-c", e["SCRIPT"]]}
if e["DC"]: d["dataCenterIds"] = [e["DC"]]
if e["VOL"]: d["networkVolumeId"] = e["VOL"]; d["volumeMountPath"] = "/vol"
print(json.dumps(d))
PY
)"
  local body; body="$(rp POST /pods "$payload")"; HTTP="$(http)"
  if [ "${HTTP#2}" = "$HTTP" ]; then echo "CREATE FAILED http $HTTP: $(printf '%s' "$body" | tr -d '\n' | cut -c1-300)"; return 1; fi
  local id cost dc
  id="$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')"
  cost="$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("costPerHr",""))')"
  dc="$(printf '%s' "$body" | python3 -c 'import json,sys; d=json.load(sys.stdin); m=d.get("machine") or {}; print(m.get("dataCenterId") or d.get("dataCenterId") or "")')"
  [ -f "$LEDGER" ] || printf '# created_at\tid\tname\tflavor\tvcpu\tcloud\tdc\tcost\tstatus\n' > "$LEDGER"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tcreated\n' "$(date -u +%FT%TZ)" "$id" "$NAME" "$FLAVOR" "$VCPU" "$CLOUD" "$dc" "$cost" >> "$LEDGER"
  echo "CREATED $id $NAME ${FLAVOR}x${VCPU} $CLOUD dc=$dc \$$cost/h"
}

cmd_get() { rp GET "/pods/$1"; echo; }

cmd_addr() { # -> "IP PORT" or empty; cached in state/addr_cache.tsv (used when the API is down)
  local id="$1" a
  a="$(rp GET "/pods/$id" 2>/dev/null | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin); ip=d.get("publicIp") or ""; pm=d.get("portMappings") or {}
    p=pm.get("22") or pm.get(22) or ""
    if ip and p: print(ip,p)
except Exception: pass
' 2>/dev/null)"
  if [ -n "$a" ]; then
    { grep -v "^$id	" "$STATE/addr_cache.tsv" 2>/dev/null; printf '%s\t%s\n' "$id" "$a"; } > "$STATE/.addr_cache.tmp" && mv "$STATE/.addr_cache.tmp" "$STATE/addr_cache.tsv"
    echo "$a"
  else
    grep "^$id	" "$STATE/addr_cache.tsv" 2>/dev/null | cut -f2
  fi
}

cmd_ssh() {
  local id="$1"; shift
  local a; a="$(cmd_addr "$id")"; [ -n "$a" ] || { echo "no address yet for $id" >&2; return 1; }
  local ip port; read -r ip port <<<"$a"
  exec ssh -i "$SSHKEY" -p "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
       -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=6 -o BatchMode=yes -o LogLevel=ERROR "root@$ip" "$@"
}

cmd_list() { rp GET /pods | python3 "$HERE/rp_list.py"; }

cmd_delete() {
  local id="$1" body
  body="$(rp DELETE "/pods/$id")"; echo "DELETE $id -> http $(http)"
  sleep 4
  rp GET "/pods/$id" >/dev/null; if [ "$(http)" = 200 ]; then echo "STILL PRESENT $id"; return 1; fi
  echo "gone $id"
  [ -f "$LEDGER" ] && { awk -F'\t' -v OFS='\t' -v id="$id" '$2==id{$9="deleted"}1' "$LEDGER" > "$LEDGER.tmp" && mv "$LEDGER.tmp" "$LEDGER"; }
  return 0
}

case "${1:-}" in
  create) shift; cmd_create "$@";;
  get) cmd_get "$2";;
  addr) cmd_addr "$2";;
  ssh) shift; cmd_ssh "$@";;
  list) cmd_list;;
  delete) cmd_delete "$2";;
  ledger) cat "$LEDGER";;
  rebuild-ledger) printf '# created_at\tid\tname\tflavor\tvcpu\tcloud\tdc\tcost\tstatus\n' > "$LEDGER"
     rp GET /pods | python3 -c '
import json,sys
d=json.load(sys.stdin); d=d if isinstance(d,list) else d.get("pods",d.get("data",[]))
for p in sorted(d,key=lambda p:p.get("createdAt","")):
    if p.get("desiredStatus")!="RUNNING" or not str(p.get("name","")).startswith("fbsim3-"): continue
    m=p.get("machine") or {}; dc=m.get("dataCenterId","") if isinstance(m,dict) else ""
    print("\t".join([p.get("createdAt","")[:19].replace(" ","T")+"Z",p["id"],p.get("name",""),p.get("cpuFlavorId",""),str(p.get("vcpuCount","")),"?",dc,str(p.get("costPerHr","")),"created"]))' >> "$LEDGER"; cat "$LEDGER";;
  *) sed -n 2,5p "$0"; exit 2;;
esac
