#!/usr/bin/env bash
# rebalance.sh [MIN_TODO] — move todo tasks from the pods with the most remaining work to pods
# whose todo has dropped below MIN_TODO (default 2 x lanes). Tasks are moved atomically on the
# source pod (mv todo -> moved) before being created on the destination, so no lane can claim twice.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
SSHKEY="$HOME/.ssh/id_ed25519_runpod"
tmp="$(mktemp -d)"
# 1. snapshot todo counts + lanes per worker
while IFS=$'\t' read -r ts id name flavor vcpu cloud dc cost status; do
  case "$ts" in \#*) continue;; esac; [ "$status" = deleted ] && continue
  case "$id" in 3xwo7xrkdsq40e|89spdc0ezuuqwi|mks2924lqlbh6a) continue;; esac
  ( t="$(bash rp.sh ssh "$id" 'echo $(ls /workspace/fleet/queue/todo | wc -l) $(ls /workspace/fleet/queue/inprog | wc -l)' 2>/dev/null)" && echo "$id $vcpu $t" > "$tmp/$id" ) &
done < state/pods.tsv; wait
cat "$tmp"/* | sort -k3 -n > "$tmp/all"
echo "== todo snapshot (id vcpu todo inprog):"; cat "$tmp/all"
# 2. donors = top pods by todo; receivers = todo < threshold
while read -r id vcpu todo inprog; do
  lanes=$((vcpu*3/4)); thr="${1:-$((lanes*2))}"
  [ "$todo" -lt "$thr" ] || continue
  need=$((lanes*3 - todo))
  donor="$(sort -k3 -n -r "$tmp/all" | head -n 1)"; read -r did dv dtodo dinprog <<<"$donor"
  dl=$((dv*3/4)); avail=$((dtodo - dl*${DONOR_FLOOR_MULT:-3})); [ "$avail" -gt 0 ] || { echo "no donor has surplus for $id"; continue; }
  k=$(( need < avail ? need : avail )); [ "$k" -gt 0 ] || continue
  echo "-- move $k tasks $did (todo $dtodo) -> $id (todo $todo)"
  # take from donor atomically
  moved="$(bash rp.sh ssh "$did" "mkdir -p /workspace/fleet/queue/moved; for f in \$(ls /workspace/fleet/queue/todo | shuf -n $k); do mv /workspace/fleet/queue/todo/\$f /workspace/fleet/queue/moved/\$f 2>/dev/null && echo \"\$f \$(cat /workspace/fleet/queue/moved/\$f)\"; done" </dev/null)"
  n="$(echo "$moved" | grep -c .)"
  [ "$n" -gt 0 ] || continue
  # create on receiver
  echo "$moved" | bash rp.sh ssh "$id" 'while read -r f a r; do [ -n "$f" ] || continue; printf "%s %s\n" "$a" "$r" > /workspace/fleet/queue/todo/$f; done; echo received $(ls /workspace/fleet/queue/todo | wc -l)'
  echo "$moved" | awk -v d="$did" -v r="$id" '{print d, r, $1}' >> state/rebalance.log
  # update snapshot numbers
  awk -v d="$did" -v r="$id" -v n="$n" '$1==d{$3-=n} $1==r{$3+=n} {print}' "$tmp/all" > "$tmp/all2" && mv "$tmp/all2" "$tmp/all"
done < "$tmp/all"
rm -rf "$tmp"
