"""Plots for the Track A free-play A/B: per-arm final gold + paired-by-seed lines."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="tmp/uplift/agentic_ab.png")
    args = ap.parse_args()
    data = json.loads(Path(args.results).read_text())
    games = [g for g in data["games"] if g.get("final_gold") is not None]
    arms = [a for a in ["control", "forecast", "scrambled"]
            if any(g["arm"] == a for g in games)]
    colors = {"control": "#888", "forecast": "#2a7", "scrambled": "#c73"}

    by_arm = defaultdict(list)
    by_seed_arm = defaultdict(lambda: defaultdict(list))
    for g in games:
        by_arm[g["arm"]].append(g["final_gold"])
        by_seed_arm[g["seed"]][g["arm"]].append(g["final_gold"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: per-arm distribution (strip + mean)
    for i, arm in enumerate(arms):
        vals = by_arm[arm]
        xs = [i + 0.04 * ((j % 5) - 2) for j in range(len(vals))]
        ax1.scatter(xs, vals, color=colors.get(arm, "#555"), alpha=0.6, s=40)
        ax1.hlines(statistics.mean(vals), i - 0.25, i + 0.25,
                   color="black", lw=2)
    ax1.set_xticks(range(len(arms)))
    ax1.set_xticklabels(arms)
    ax1.set_ylabel("Final gold")
    ax1.set_title("Final gold by arm (black = mean)")

    # Right: paired-by-seed lines control -> forecast
    seeds = sorted(by_seed_arm)
    for seed in seeds:
        am = by_seed_arm[seed]
        pts = [(a, statistics.mean(am[a])) for a in arms if am.get(a)]
        if len(pts) >= 2:
            ax2.plot([p[0] for p in pts], [p[1] for p in pts],
                     marker="o", label=f"seed{seed}")
    ax2.set_ylabel("Mean final gold (per seed)")
    ax2.set_title("Paired by seed")
    ax2.legend(fontsize=8)

    fig.suptitle("Track A: forecast-informed vs control free-play (gold)")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
