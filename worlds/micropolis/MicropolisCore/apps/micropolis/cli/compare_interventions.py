#!/usr/bin/env python3
"""Compare intervention sweeps against matched-seed baselines.

For each intervention file, pairs runs by seed with the baseline runs of the
same city and reports, over the post-intervention window, the change in event
probabilities and in end-of-window population quantiles.

Usage:
  python3 cli/compare_interventions.py --baseline 'sweeps/baseline-*.jsonl' \
      --interventions 'sweeps/iv-*.jsonl' --window-start 2 --window-end 12
"""

import argparse
import glob
import json
from collections import defaultdict

import analyze_sweep as A

TURNS_PER_YEAR = 48


def load_by_city(patterns):
    runs = defaultdict(dict)
    for pattern in patterns:
        for p in sorted(glob.glob(pattern)):
            with open(p) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        runs[r["city"]][r["seed"]] = r
    return runs


def window_events(run, t0, t1):
    out = set()
    for e in run["events"]:
        if t0 < e["turn"] <= t1:
            name = A.MESSAGE_NAMES.get(e["msg"]) if isinstance(e["msg"], int) else e["msg"]
            if name:
                out.add(name)
    for name, turn in A.derived_event_turns(run).items():
        if t0 < turn <= t1:
            out.add(name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--interventions", nargs="+", required=True)
    ap.add_argument("--window-start", type=float, default=2)
    ap.add_argument("--window-end", type=float, default=12)
    ap.add_argument("--min-delta", type=float, default=0.15)
    args = ap.parse_args()

    base = load_by_city(args.baseline)
    t0 = round(args.window_start * TURNS_PER_YEAR)
    t1 = round(args.window_end * TURNS_PER_YEAR)

    for pattern in args.interventions:
        for p in sorted(glob.glob(pattern)):
            runs = []
            with open(p) as f:
                for line in f:
                    if line.strip():
                        runs.append(json.loads(line))
            if not runs:
                continue
            city = runs[0]["city"]
            iv = runs[0]["interventions"]
            paired = [(r, base[city].get(r["seed"])) for r in runs]
            paired = [(r, b) for r, b in paired if b]
            n = len(paired)
            if not n:
                print(f"\n## {p}: no matched baseline for {city}")
                continue

            print(f"\n## {city} + {json.dumps(iv)}   vs baseline, window y{args.window_start:g}..y{args.window_end:g}  (n={n})")

            names = set()
            iv_ev, ba_ev = [], []
            for r, b in paired:
                ei, eb = window_events(r, t0, t1), window_events(b, t0, t1)
                iv_ev.append(ei)
                ba_ev.append(eb)
                names |= ei | eb
            rows = []
            for name in sorted(names):
                pi = sum(1 for e in iv_ev if name in e) / n
                pb = sum(1 for e in ba_ev if name in e) / n
                if abs(pi - pb) >= args.min_delta:
                    rows.append((name, pb, pi))
            if rows:
                print(f"  {'event in window':<26}{'base':>7}{'interv':>8}{'delta':>8}")
                for name, pb, pi in sorted(rows, key=lambda r: -abs(r[2] - r[1])):
                    print(f"  {name:<26}{pb:>7.2f}{pi:>8.2f}{pi-pb:>+8.2f}")
            else:
                print(f"  (no event-probability deltas >= {args.min_delta})")

            for metric in ("cityPop", "totalFunds", "cityScore", "pollutionAverage"):
                iv_vals = sorted(A.row_at_year(r, args.window_end)[metric] for r, _ in paired)
                ba_vals = sorted(A.row_at_year(b, args.window_end)[metric] for _, b in paired)
                med_iv = A.quantile(iv_vals, 0.5)
                med_ba = A.quantile(ba_vals, 0.5)
                if med_ba or med_iv:
                    print(f"  {metric} p50 @y{args.window_end:g}: base {med_ba:.0f} -> interv {med_iv:.0f}")


if __name__ == "__main__":
    main()
