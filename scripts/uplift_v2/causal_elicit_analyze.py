#!/usr/bin/env python3
"""Score B2 causal elicitation results against simulator ground truth.

Ground truth: tmp/causal/gap_stats_seed2.json (seed2, N~180/arm) and
tmp/causal/analysis.json rows (seeds 0/1/3, N=40/arm).

Per (seed, qid): model p̂ per framing = mean over samples (parsed only).
  dhat_do  = p̂(do)  - p̂(base)        vs  Δ*_do  = p_do  - p0
  dhat_obs = p̂(obs) - p̂(base)        vs  Δ*_obs = p_obs - p0
  dhat_placebo_do / _obs             vs  0 (exact null)

Reports per model:
  1. Δ-accuracy: Pearson/Spearman corr + MAE of dhat_do vs Δ*_do; also for a
     no-update baseline (dhat=0) — MAE(Δ*_do) itself.
  2. Sign accuracy on |Δ*_do|>0.15 questions (vs 50% chance).
  3. do-vs-obs distinction: mean |dhat_do - dhat_obs|, corr(dhat_do - dhat_obs,
     Δ*_do - Δ*_obs) (the true confounding gap, sign-flipped).
  4. Placebo: mean |dhat_placebo| vs mean |dhat_do| (does the model move more
     for real interventions than for no-ops?).
  DiD metric: (dhat_do - dhat_placebo_do) - Δ*_do.

Usage:
  uv run python scripts/uplift_v2/causal_elicit_analyze.py \
      tmp/causal/elicit_q4b.json [tmp/causal/elicit_oss120.json ...]
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(order):
            r[i] = k
        return r
    return pearson(rank(xs), rank(ys))


def load_truth() -> dict:
    truth = {}
    a = json.load(open("tmp/causal/analysis.json"))
    for r in a["rows"]:
        truth[(f"seed{r['seed']}", r["qid"])] = {
            "p0": r["p_y"], "p_do": r["p_do"], "p_obs": r["p_obs"],
            "n": r["n_base"]}
    gp = Path("tmp/causal/gap_stats_seed2.json")
    if gp.exists():
        for r in json.loads(gp.read_text()):
            truth[("seed2", r["qid"])] = {
                "p0": r["p0"], "p_do": r["p_do"], "p_obs": r["p_obs"],
                "n": r["n0"]}
    return truth


def analyze(path: str, truth: dict) -> None:
    d = json.load(open(path))
    tag = d["tag"]
    byq = defaultdict(lambda: defaultdict(list))
    for r in d["results"]:
        if r.get("p") is not None:
            byq[(r["game_id"], r["qid"])][r["framing"]].append(r["p"])

    rows = []
    for key, fr in byq.items():
        if key not in truth:
            continue
        t = truth[key]
        if any(f not in fr for f in
               ("base", "do", "obs", "placebo_do", "placebo_obs")):
            continue
        m = {f: sum(v) / len(v) for f, v in fr.items()}
        rows.append({
            "key": key,
            "dhat_do": m["do"] - m["base"],
            "dhat_obs": m["obs"] - m["base"],
            "dhat_pdo": m["placebo_do"] - m["base"],
            "dhat_pobs": m["placebo_obs"] - m["base"],
            "true_do": t["p_do"] - t["p0"],
            "true_obs": (t["p_obs"] - t["p0"]) if t["p_obs"] is not None else None,
            "p_base_hat": m["base"], "p0": t["p0"],
        })

    n = len(rows)
    print(f"\n===== {tag} ({path}): {n} scored (seed,question) pairs =====")
    if not n:
        return

    dd = [r["dhat_do"] for r in rows]
    td = [r["true_do"] for r in rows]
    mae = sum(abs(a - b) for a, b in zip(dd, td)) / n
    mae0 = sum(abs(b) for b in td) / n
    print(f"1. delta-accuracy: corr(dhat_do, D*_do) pearson={pearson(dd, td):.3f} "
          f"spearman={spearman(dd, td):.3f}")
    print(f"   MAE(dhat_do vs D*_do)={mae:.4f}  vs no-update baseline {mae0:.4f} "
          f"({'BEATS' if mae < mae0 else 'LOSES TO'} no-update)")

    big = [r for r in rows if abs(r["true_do"]) > 0.15]
    if big:
        sign_ok = sum(1 for r in big
                      if r["dhat_do"] * r["true_do"] > 0)
        print(f"2. sign accuracy on |D*_do|>0.15 (n={len(big)}): "
              f"{sign_ok}/{len(big)} = {sign_ok/len(big):.2f} (chance 0.5)")

    diffs = [r["dhat_do"] - r["dhat_obs"] for r in rows]
    gaps = [r["true_do"] - r["true_obs"] for r in rows
            if r["true_obs"] is not None]
    dg = [(r["dhat_do"] - r["dhat_obs"], r["true_do"] - r["true_obs"])
          for r in rows if r["true_obs"] is not None]
    print(f"3. do-vs-obs distinction: mean|dhat_do - dhat_obs|="
          f"{sum(abs(x) for x in diffs)/n:.4f}; "
          f"corr with true (D*_do - D*_obs): "
          f"pearson={pearson([a for a,b in dg],[b for a,b in dg]):.3f} "
          f"(n={len(dg)})")

    for f, lbl in (("dhat_pdo", "placebo_do"), ("dhat_pobs", "placebo_obs")):
        vals = [abs(r[f]) for r in rows]
        print(f"4. |{lbl}| mean={sum(vals)/n:.4f}  "
              f"(vs |dhat_do| mean={sum(abs(x) for x in dd)/n:.4f})")

    did = [(r["dhat_do"] - r["dhat_pdo"]) - r["true_do"] for r in rows]
    print(f"5. DiD error mean|.|={sum(abs(x) for x in did)/n:.4f}")

    base_brier = sum((r["p_base_hat"] - r["p0"]) ** 2 for r in rows) / n
    print(f"   (base-framing Brier vs p0: {base_brier:.4f})")


def main() -> int:
    truth = load_truth()
    for path in sys.argv[1:]:
        analyze(path, truth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
