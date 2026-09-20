#!/usr/bin/env bash
# status.sh — one line per live pod + totals. Reads the ledger; skips deleted.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tot_done=0; tot_todo=0; tot_inprog=0; tot_failed=0; tot_recent=0; tot_lanes=0
while IFS=$'\t' read -r ts id name flavor vcpu cloud dc cost status; do
  case "$ts" in \#*) continue;; esac
  [ "$status" = deleted ] && continue
  line="$(bash "$HERE/rp.sh" ssh "$id" 'bash -s' < "$HERE/on_pod/status_pod.sh" 2>/dev/null)" || line="UNREACHABLE"
  printf '%-14s %-16s %3sv %s\n' "$id" "$name" "$vcpu" "$line"
  for kv in $line; do k="${kv%%=*}"; v="${kv#*=}"; case "$k" in
    done) tot_done=$((tot_done+v));; todo) tot_todo=$((tot_todo+v));; inprog) tot_inprog=$((tot_inprog+v));;
    failed) tot_failed=$((tot_failed+v));; done30m) tot_recent=$((tot_recent+v));; lanes) tot_lanes=$((tot_lanes+v));; esac; done
done < "$HERE/state/pods.tsv"
rate=$((tot_recent*2))
eta="n/a"; [ "$rate" -gt 0 ] && eta="$(( (tot_todo+tot_inprog) * 60 / rate )) min"
echo "TOTAL lanes=$tot_lanes done=$tot_done inprog=$tot_inprog todo=$tot_todo failed=$tot_failed rate=${rate}/h eta=$eta  $(date -u +%H:%MZ)"
