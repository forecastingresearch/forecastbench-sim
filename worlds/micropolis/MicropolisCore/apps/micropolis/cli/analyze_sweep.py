#!/usr/bin/env python3
"""Analyze sweep_sim.js output.

Reads one or more sweep JSONL files (one run per line) and reports, per city:

  * metric trajectory quantiles by year (population, funds, score, ...)
  * P(event by horizon) for a catalog of events, estimated across seeds
  * 2x2 joint tables for selected event pairs (dependence structure)

Events come from two sources: engine messages (disasters, warnings) captured
by the sweep's sendMessage callback, and predicates evaluated on the yearly
snapshot rows (threshold crossings, treasury exhaustion, tile-census changes).

Usage:
  python3 cli/analyze_sweep.py sweeps/baseline-*.jsonl --horizons 1 2 5 10 20 25
  python3 cli/analyze_sweep.py sweeps/*.jsonl --json summary.json
"""

import argparse
import glob
import json
import math
import sys
from collections import defaultdict

MESSAGE_NAMES = {
    1: "need_res", 2: "need_com", 3: "need_ind", 4: "need_roads",
    5: "need_rails", 6: "need_power", 7: "need_stadium", 8: "need_seaport",
    9: "need_airport", 10: "high_pollution", 11: "high_crime",
    12: "traffic_jams", 13: "need_fire_st", 14: "need_police_st",
    15: "blackouts", 16: "tax_too_high", 17: "road_needs_funding",
    18: "fire_needs_funding", 19: "police_needs_funding", 20: "fire",
    21: "monster", 22: "tornado", 23: "earthquake", 24: "plane_crash",
    25: "ship_crash", 26: "train_crash", 27: "heli_crash",
    28: "high_unemployment", 29: "city_broke", 30: "firebombing",
    31: "need_parks", 32: "explosion", 35: "reached_town", 36: "reached_city",
    37: "reached_capital", 38: "reached_metropolis", 39: "reached_megalopolis",
    40: "brownouts", 41: "heavy_traffic", 42: "flood", 43: "meltdown",
    44: "riots",
}

TURNS_PER_YEAR = 48


def load_runs(paths):
    runs = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    runs.append(json.loads(line))
    return runs


def first_message_turns(run):
    """Turn of first occurrence for each named message in a run."""
    first = {}
    for e in run["events"]:
        name = MESSAGE_NAMES.get(e["msg"]) if isinstance(e["msg"], int) else e["msg"]
        if name and name not in first:
            first[name] = e["turn"]
    return first


def row_at_year(run, year):
    """Snapshot row at (or nearest before) a given year offset from run start."""
    target = round(year * TURNS_PER_YEAR)
    best = None
    for r in run["rows"]:
        if r["turn"] <= target:
            best = r
        else:
            break
    return best


def derived_event_turns(run):
    """First-occurrence turns of events derived from yearly snapshot rows.

    Yearly sampling means the turn is an upper bound (rounded up to the next
    sample), which is fine for year-granularity horizons.
    """
    rows = run["rows"]
    base = rows[1] if len(rows) > 1 else rows[0]  # rows[0] predates first eval
    first = {}

    def hit(name, turn):
        if name not in first:
            first[name] = turn

    peak_pop = 0
    for r in rows[1:]:
        t = r["turn"]
        peak_pop = max(peak_pop, r["cityPop"])
        if r["totalFunds"] <= 0:
            hit("funds_zero", t)
        if not r["autoBudget"]:
            hit("autobudget_off", t)
        if r["roadEffect"] < 32:
            hit("road_underfunded", t)
        if r["policeEffect"] < 700:
            hit("police_under_70pct", t)
        if r["fireEffect"] < 700:
            hit("fire_under_70pct", t)
        if r["pollutionAverage"] > 60:
            hit("pollution_gt60", t)
        if r["crimeAverage"] > 100:
            hit("crime_gt100", t)
        if r["trafficAverage"] > 60:
            hit("traffic_gt60", t)
        if r["unpoweredZoneCount"] > 0:
            hit("unpowered_zones", t)
        if r["unpoweredZoneCount"] >= 20:
            hit("unpowered_20plus", t)
        if r["census"]["rad"] > 0:
            hit("radioactive_tiles", t)
        if r["census"]["flood"] > 0:
            hit("flood_tiles", t)
        if r["census"]["fire"] > 0:
            hit("fire_tiles", t)
        if r["census"]["rubble"] >= base["census"]["rubble"] + 20:
            hit("rubble_plus20", t)
        if r["nuclearPowerPop"] < base["nuclearPowerPop"]:
            hit("lost_nuclear_plant", t)
        if r["coalPowerPop"] < base["coalPowerPop"]:
            hit("lost_coal_plant", t)
        if r["stadiumPop"] < base["stadiumPop"]:
            hit("lost_stadium", t)
        if r["airportPop"] < base["airportPop"]:
            hit("lost_airport", t)
        if r["seaportPop"] < base["seaportPop"]:
            hit("lost_seaport", t)
        if r["hospitalPop"] > base["hospitalPop"]:
            hit("gained_hospital", t)
        if r["hospitalPop"] < base["hospitalPop"]:
            hit("lost_hospital", t)
        if r["churchPop"] > base["churchPop"]:
            hit("gained_church", t)
        if r["cityClass"] > base["cityClass"]:
            hit("class_up", t)
        if r["cityClass"] < base["cityClass"]:
            hit("class_down", t)
        if base["cityPop"] > 0 and r["cityPop"] <= 0.9 * base["cityPop"]:
            hit("pop_down_10pct", t)
        if base["cityPop"] > 0 and r["cityPop"] <= 0.75 * base["cityPop"]:
            hit("pop_down_25pct", t)
        if base["cityPop"] > 0 and r["cityPop"] >= 1.1 * base["cityPop"]:
            hit("pop_up_10pct", t)
        if peak_pop > 0 and r["cityPop"] <= 0.85 * peak_pop and r["turn"] >= 96:
            hit("pop_15pct_off_peak", t)
        if r["cityScore"] < 500:
            hit("score_below_500", t)
        if r["census"]["road"] <= base["census"]["road"] - 50:
            hit("lost_50_road_tiles", t)
        if r["resCap"]:
            hit("res_capped", t)
    return first


def all_event_turns(run):
    ev = first_message_turns(run)
    for k, v in derived_event_turns(run).items():
        if k not in ev or v < ev[k]:
            ev[k] = v
    return ev


def pair_table(events_per_run, a, ah, b, bh):
    """2x2 joint counts of (A by year ah) x (B by year bh) across runs."""
    n = [[0, 0], [0, 0]]
    for ev in events_per_run:
        got_a = ev.get(a, 1 << 30) <= ah * TURNS_PER_YEAR
        got_b = ev.get(b, 1 << 30) <= bh * TURNS_PER_YEAR
        n[1 if got_a else 0][1 if got_b else 0] += 1
    return n


def print_pairs(label, events_per_run, pairs):
    total = len(events_per_run)
    print(f"\n--- joint tables: {label} (n={total}) ---")
    print(f"{'A':<28}{'B':<28}{'P(A)':>6}{'P(B)':>6}{'P(A&B)':>8}{'P(B|A)':>8}{'P(B|~A)':>8}{'indep':>7}")
    for a, ah, b, bh in pairs:
        n = pair_table(events_per_run, a, ah, b, bh)
        na = n[1][0] + n[1][1]
        nb = n[0][1] + n[1][1]
        pa, pb = na / total, nb / total
        pab = n[1][1] / total
        pba = n[1][1] / na if na else float("nan")
        pbna = n[0][1] / (total - na) if total - na else float("nan")
        print(f"{a}@y{ah:<24g}{b}@y{bh:<24g}"[:56].ljust(56) +
              f"{pa:>6.2f}{pb:>6.2f}{pab:>8.2f}{pba:>8.2f}{pbna:>8.2f}{pa*pb:>7.2f}")


def quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = q * (len(sorted_vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--horizons", nargs="*", type=float, default=[1, 2, 5, 10, 15, 20, 25])
    ap.add_argument("--metrics", nargs="*", default=["cityPop", "totalFunds", "cityScore", "pollutionAverage", "crimeAverage", "landValueAverage"])
    ap.add_argument("--json", help="Write full summary JSON here")
    ap.add_argument("--min-rate", type=float, default=0.0, help="Only print events with max rate >= this")
    ap.add_argument("--pairs", action="store_true", help="Print 2x2 joint tables for a built-in list of event pairs")
    ap.add_argument("--no-rates", action="store_true", help="Skip the per-event rate tables")
    args = ap.parse_args()

    paths = []
    for pattern in args.files:
        paths.extend(sorted(glob.glob(pattern)))
    runs = load_runs(paths)
    if not runs:
        sys.exit("no runs loaded")

    by_city = defaultdict(list)
    for run in runs:
        key = (run["city"], json.dumps(run.get("interventions", [])))
        by_city[key].append(run)

    summary = {}
    for (city, iv), city_runs in sorted(by_city.items()):
        n = len(city_runs)
        label = city if iv == "[]" else f"{city} +{iv}"
        events = [all_event_turns(r) for r in city_runs]
        names = sorted({k for ev in events for k in ev})
        rates = {}
        for name in names:
            rates[name] = {
                str(h): sum(1 for ev in events if ev.get(name, 1 << 30) <= h * TURNS_PER_YEAR) / n
                for h in args.horizons
            }

        metrics = {}
        for m in args.metrics:
            metrics[m] = {}
            for h in args.horizons:
                vals = sorted(
                    r_at[m] for r_at in (row_at_year(run, h) for run in city_runs) if r_at is not None
                )
                metrics[m][str(h)] = {
                    "p10": quantile(vals, 0.10),
                    "p50": quantile(vals, 0.50),
                    "p90": quantile(vals, 0.90),
                }

        summary[label] = {"n": n, "eventRates": rates, "metrics": metrics}

        if not args.no_rates:
            print(f"\n=== {label}  (n={n}) ===")
            header = "event".ljust(24) + "".join(f"y{h:g}".rjust(7) for h in args.horizons)
            print(header)
            for name in names:
                row = rates[name]
                if max(row.values()) < args.min_rate:
                    continue
                print(name.ljust(24) + "".join(f"{row[str(h)]:7.2f}" for h in args.horizons))
            for m in args.metrics:
                print(m + " p50: " + "  ".join(f"y{h:g}={metrics[m][str(h)]['p50']:.0f}" for h in args.horizons if metrics[m][str(h)]['p50'] is not None))

        if args.pairs:
            PAIRS = [
                ("earthquake", 10, "fire_tiles", 10),
                ("earthquake", 10, "rubble_plus20", 10),
                ("earthquake", 10, "pop_down_10pct", 15),
                ("tornado", 10, "flood", 10),
                ("tornado", 10, "earthquake", 10),
                ("meltdown", 20, "radioactive_tiles", 20),
                ("meltdown", 20, "pop_down_10pct", 25),
                ("monster", 10, "pollution_gt60", 10),
                ("plane_crash", 5, "heli_crash", 5),
                ("plane_crash", 5, "fire_tiles", 5),
                ("blackouts", 10, "class_down", 15),
                ("unpowered_20plus", 10, "pop_down_25pct", 15),
                ("flood", 10, "earthquake", 10),
                ("city_broke", 10, "pop_down_10pct", 15),
                ("lost_coal_plant", 10, "unpowered_20plus", 10),
                ("lost_nuclear_plant", 20, "meltdown", 20),
                ("ship_crash", 10, "lost_seaport", 15),
                ("gained_hospital", 10, "pop_up_10pct", 10),
            ]
            print_pairs(label, events, PAIRS)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
