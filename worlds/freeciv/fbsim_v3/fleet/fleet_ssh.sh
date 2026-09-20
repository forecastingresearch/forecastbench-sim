#!/usr/bin/env bash
# fleet_ssh.sh [--workers] CMD — run CMD on every live pod (parallel), print "id name: output". --workers excludes archive/vol pods.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
only_workers=0; [ "${1:-}" = "--workers" ] && { only_workers=1; shift; }
CMD="$1"; tmp="$(mktemp -d)"
while IFS=$'\t' read -r ts id name flavor vcpu cloud dc cost status; do
  case "$ts" in \#*) continue;; esac; [ "$status" = deleted ] && continue
  if [ "$only_workers" = 1 ]; then case "$id" in 3xwo7xrkdsq40e|89spdc0ezuuqwi|mks2924lqlbh6a) continue;; esac; fi
  ( out="$(bash "$HERE/rp.sh" ssh "$id" "$CMD" 2>&1)" || out="UNREACHABLE: $out"; printf '%s %s: %s\n' "$id" "$name" "$(echo "$out" | tr '\n' '|' | cut -c1-300)" > "$tmp/$id" ) &
done < "$HERE/state/pods.tsv"
wait; cat "$tmp"/* | sort -k2; rm -rf "$tmp"
