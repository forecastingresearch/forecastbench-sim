"""Track A analysis: does the forecast-informed agent accumulate more gold?

Loads agentic_ab.json, reports per-arm final-gold stats and a paired (by seed)
control-vs-forecast comparison with bootstrap CI and Wilcoxon.

Usage:
  PYTHONPATH=src uv run python scripts/uplift/agentic_analyze.py \
    --results data/uplift/agentic_ab.json
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


def bootstrap_ci(diffs, n=10000, seed=0):
    if not diffs:
        return (None, None)
    rng = np.random.default_rng(seed)
    arr = np.array(diffs, float)
    means = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon(diffs):
    try:
        from scipy.stats import wilcoxon as w
        nz = [d for d in diffs if d != 0]
        return float(w(nz).pvalue) if len(nz) >= 3 else None
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.results).read_text())
    games = [g for g in data["games"] if g.get("final_gold") is not None]

    by_arm = defaultdict(list)
    by_seed_arm = defaultdict(lambda: defaultdict(list))
    for g in games:
        by_arm[g["arm"]].append(g["final_gold"])
        by_seed_arm[g["seed"]][g["arm"]].append(g["final_gold"])

    print("=" * 66)
    print("TRACK A — free-play final gold by arm")
    print("=" * 66)
    for arm, vals in by_arm.items():
        print(f"  {arm:10s}: mean={statistics.mean(vals):7.1f} "
              f"median={statistics.median(vals):7.1f} "
              f"min={min(vals):.0f} max={max(vals):.0f} n={len(vals)}")
    print()

    # Paired by seed (mean over repeats per arm).
    print("Paired by seed (mean gold per arm):")
    diffs = []
    for seed in sorted(by_seed_arm):
        am = by_seed_arm[seed]
        if am.get("control") and am.get("forecast"):
            c = statistics.mean(am["control"])
            f = statistics.mean(am["forecast"])
            diffs.append(f - c)
            print(f"  seed{seed}: control={c:.0f}  forecast={f:.0f}  diff={f-c:+.0f}")
    if diffs:
        lo, hi = bootstrap_ci(diffs)
        p = wilcoxon(diffs)
        md = statistics.mean(diffs)
        sig = "  *95% CI excludes 0*" if (lo is not None and (lo > 0 or hi < 0)) else ""
        print(f"\n  forecast - control: mean gold diff = {md:+.1f}  "
              f"95%CI=[{lo:+.1f},{hi:+.1f}]  Wilcoxon p="
              f"{p if p is None else round(p,4)}  n_seed={len(diffs)}{sig}")

    # scrambled comparison if present
    if any(g["arm"] == "scrambled" for g in games):
        sdiffs = []
        for seed in sorted(by_seed_arm):
            am = by_seed_arm[seed]
            if am.get("forecast") and am.get("scrambled"):
                sdiffs.append(statistics.mean(am["forecast"]) - statistics.mean(am["scrambled"]))
        if sdiffs:
            lo, hi = bootstrap_ci(sdiffs)
            print(f"\n  forecast - scrambled: mean diff = {statistics.mean(sdiffs):+.1f} "
                  f"95%CI=[{lo:+.1f},{hi:+.1f}] n={len(sdiffs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
