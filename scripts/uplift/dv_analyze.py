"""Track B analysis: does giving the LLM game-true forecasts raise gold?

Loads the decisions file (dv_decide.py) + bank (dv_build_bank.py) and reports,
per arm (control / true / scrambled):
  - mean gold of the chosen government (primary outcome)
  - mean regret vs the best government, best-pick rate
  - choice distribution
Plus paired scenario-level comparisons (true vs control, true vs scrambled) with
bootstrap CIs and a Wilcoxon test, a forecast sanity report, and a Value-of-
Information manipulation check (does the best government actually vary, and do
forecasts track it?).

Usage:
  PYTHONPATH=src uv run python scripts/uplift/dv_analyze.py \
    --bank data/uplift/bank_pilot.json --decisions data/uplift/decisions_pilot.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

GOVERNMENTS = ["Despotism", "Monarchy", "Republic", "Democracy"]


def bootstrap_ci(diffs, n=10000, seed=0):
    if not diffs:
        return (None, None)
    rng = np.random.default_rng(seed)
    arr = np.array(diffs, dtype=float)
    means = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def wilcoxon(diffs):
    try:
        from scipy.stats import wilcoxon as w
        nz = [d for d in diffs if d != 0]
        if len(nz) < 3:
            return None
        return float(w(nz).pvalue)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--decisions", required=True)
    args = ap.parse_args()

    bank = json.loads(Path(args.bank).read_text())
    dec = json.loads(Path(args.decisions).read_text())
    scenarios = {(s["seed"], s["checkpoint"]): s for s in bank["scenarios"]}
    decisions = [d for d in dec["decisions"] if d.get("choice")]
    arms = dec["arms"]

    print("=" * 70)
    print("FORECAST SANITY (game-true p_mc across scenarios)")
    print("=" * 70)
    ev_names = bank.get("event_names", [])
    last_h = f"h{bank['scenarios'][0]['fc_horizons'][-1]}"
    for e in ev_names:
        vals = [s["p_mc"][last_h][e] for s in bank["scenarios"]
                if s["p_mc"][last_h].get(e) is not None]
        if vals:
            print(f"  {e:14s} {last_h}: min={min(vals):.2f} max={max(vals):.2f} "
                  f"mean={statistics.mean(vals):.2f} "
                  f"(spread {max(vals)-min(vals):.2f})")
    print()

    print("=" * 70)
    print("VALUE-OF-INFORMATION MANIPULATION CHECK")
    print("=" * 70)
    best = [s["best_gov"] for s in bank["scenarios"] if s["best_gov"]]
    spreads = [s["gold_spread"] for s in bank["scenarios"] if s.get("gold_spread")]
    dist = {g: best.count(g) for g in GOVERNMENTS}
    print(f"  best-government distribution across scenarios: {dist}")
    print(f"  gold spread (best-worst gov) mean={statistics.mean(spreads):.0f} "
          f"min={min(spreads):.0f} max={max(spreads):.0f}" if spreads else "  no spreads")
    print(f"  => VOI exists only if best-gov varies AND spread is large enough "
          f"to matter. Distinct best-govs: {len([g for g,c in dist.items() if c])}")
    print()

    print("=" * 70)
    print("OUTCOMES BY ARM")
    print("=" * 70)
    by_arm = defaultdict(list)      # arm -> list of gold outcomes (all reps)
    by_arm_regret = defaultdict(list)
    by_arm_best = defaultdict(list)
    by_arm_choice = defaultdict(lambda: defaultdict(int))
    scen_arm_gold = defaultdict(lambda: defaultdict(list))  # (seed,ckpt)->arm->[gold]
    for d in decisions:
        o = d.get("outcome")
        if not o:
            continue
        arm = d["arm"]
        by_arm[arm].append(o["gold"])
        by_arm_regret[arm].append(o["regret"])
        by_arm_best[arm].append(1 if o["chose_best"] else 0)
        by_arm_choice[arm][d["choice"]] += 1
        scen_arm_gold[(d["seed"], d["ckpt"])][arm].append(o["gold"])

    for arm in arms:
        g = by_arm[arm]
        if not g:
            print(f"  {arm:10s}: no data")
            continue
        print(f"  {arm:10s}: mean gold={statistics.mean(g):7.1f}  "
              f"regret={statistics.mean(by_arm_regret[arm]):6.1f}  "
              f"best-pick={statistics.mean(by_arm_best[arm])*100:4.0f}%  "
              f"n={len(g)}  choices={dict(by_arm_choice[arm])}")
    print()

    print("=" * 70)
    print("PAIRED SCENARIO-LEVEL COMPARISONS (mean gold per scenario)")
    print("=" * 70)

    def paired(a, b):
        diffs = []
        for key, am in scen_arm_gold.items():
            if am.get(a) and am.get(b):
                diffs.append(statistics.mean(am[a]) - statistics.mean(am[b]))
        return diffs

    for a, b in [("true", "control"), ("true", "scrambled"), ("scrambled", "control")]:
        if a in arms and b in arms:
            diffs = paired(a, b)
            if not diffs:
                continue
            lo, hi = bootstrap_ci(diffs)
            p = wilcoxon(diffs)
            md = statistics.mean(diffs)
            sig = ""
            if lo is not None and (lo > 0 or hi < 0):
                sig = "  *95% CI excludes 0*"
            print(f"  {a} - {b}: mean gold diff = {md:+.1f}  "
                  f"95%CI=[{lo:+.1f},{hi:+.1f}]  "
                  f"Wilcoxon p={p if p is None else round(p,4)}  n_scen={len(diffs)}{sig}")
    print()
    print("Interpretation: 'true - control' > 0 means true forecasts raised gold; "
          "'true - scrambled' > 0 isolates value of the TRUE probabilities from "
          "merely having numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
