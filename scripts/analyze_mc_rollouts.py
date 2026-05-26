#!/usr/bin/env python3
"""Analyze the Monte Carlo rollout outputs from monte_carlo_rollout.py.

Usage:
    uv run python scripts/analyze_mc_rollouts.py \\
        tmp/mc/method_b_n10.json tmp/mc/method_b_n10_batch2.json \\
        --threshold 155 --player 1
"""
from __future__ import annotations

import sys
import json
import math
import argparse
from pathlib import Path


def load(path: str) -> tuple[str, list[dict]]:
    data = json.loads(Path(path).read_text())
    return data["config"]["method"], [r for r in data["results"] if r["success"]]


def player_scores(results: list[dict], player_id: int) -> list[int]:
    out = []
    for r in results:
        ps = r.get("player_states", {})
        s = ps.get(str(player_id), {}).get("score")
        if s is None or s < 0:
            continue
        out.append(int(s))
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson 95% CI for Bernoulli(p)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def summarize(scores: list[int]) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0}
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / max(1, n - 1)
    return {
        "n": n,
        "min": min(scores),
        "max": max(scores),
        "mean": mean,
        "std": math.sqrt(var),
        "sorted": sorted(scores),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="JSON files produced by monte_carlo_rollout.py")
    ap.add_argument("--player", type=int, default=1)
    ap.add_argument("--threshold", type=int, required=True)
    args = ap.parse_args()

    all_scores: list[int] = []
    for p in args.inputs:
        method, results = load(p)
        scores = player_scores(results, args.player)
        summ = summarize(scores)
        k = sum(1 for s in scores if s >= args.threshold)
        p_hat, lo, hi = wilson(k, len(scores))
        print(f"== {p} (method={method}) ==")
        print(f"   player {args.player} scores: {summ.get('sorted')}")
        print(f"   n={summ['n']}  mean={summ['mean']:.2f}  std={summ['std']:.2f}  "
              f"range=[{summ.get('min')}, {summ.get('max')}]")
        print(f"   P(score >= {args.threshold}) = {k}/{summ['n']} = {p_hat:.2f}  "
              f"(Wilson 95% CI [{lo:.2f}, {hi:.2f}])")
        print()
        all_scores.extend(scores)

    if len(args.inputs) > 1:
        summ = summarize(all_scores)
        k = sum(1 for s in all_scores if s >= args.threshold)
        p_hat, lo, hi = wilson(k, len(all_scores))
        print(f"== Pooled ==")
        print(f"   n={summ['n']}  mean={summ['mean']:.2f}  std={summ['std']:.2f}")
        print(f"   P(score >= {args.threshold}) = {k}/{summ['n']} = {p_hat:.2f}  "
              f"(Wilson 95% CI [{lo:.2f}, {hi:.2f}])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
