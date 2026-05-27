#!/usr/bin/env python3
"""Compare LLM rankings: Brier vs hard 0/1 label vs Brier vs dense MC probability.

Takes:
  - an LLM eval run JSON (from evaluate_llm_forecasts_parallel.py) with per-question
    per-model predicted probabilities and the canonical 0/1 ground_truth
  - one or more mc_resolve outputs providing p_mc[question_id]

For each model, computes:
  - brier_hard  = mean (pred - y01)^2          over questions with a p_mc
  - brier_dense = mean (pred - p_mc)^2
Then ranks models under each and reports Spearman correlation + rank shifts.

Note: brier_dense is not directly comparable in level to brier_hard (different
targets); the object of interest is the *ranking*, and the per-model gap to the
best achievable score (a forecaster predicting p_mc exactly).

Usage:
    uv run python scripts/score_mc_vs_binary.py \\
        --eval tmp/mc_eval/eval.json \\
        --mc tmp/mc_resolve/seed0_h1.json tmp/mc_resolve/seed1_h1.json \\
        --output tmp/mc_eval/ranking_comparison.json
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path


def spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    vy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx * vy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="LLM eval run JSON")
    ap.add_argument("--mc", nargs="+", required=True, help="mc_resolve output JSON(s)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-rollouts", type=int, default=8,
                    help="Only use questions with at least this many successful rollouts")
    args = ap.parse_args()

    # Merge p_mc + baseline (canonical single-future 0/1) across games.
    # IMPORTANT: question_ids (q0000...) are per-game, NOT global, so we key
    # everything by (game_id, question_id). game_id comes from each mc file's
    # config. We take y01 from the no-reseed baseline rollout so the hard label
    # and the dense p_mc come from the *identical* fork simulator config.
    p_mc: dict[tuple, float] = {}
    n_roll: dict[tuple, int] = {}
    base01: dict[tuple, float] = {}
    for path in args.mc:
        d = json.loads(Path(path).read_text())
        gid = d.get("config", {}).get("game_id", "?")
        for qid, p in d.get("p_mc", {}).items():
            p_mc[(gid, qid)] = p
            n_roll[(gid, qid)] = d.get("n_rollouts", {}).get(qid, 0)
        for qid, ans in (d.get("baseline") or {}).items():
            if ans is not None:
                base01[(gid, qid)] = 1.0 if ans else 0.0

    usable = {k for k, n in n_roll.items() if n >= args.min_rollouts}

    ev = json.loads(Path(args.eval).read_text())
    questions = ev["questions"] if isinstance(ev, dict) and "questions" in ev else ev

    # Collect per-model squared errors
    per_model = {}  # model -> {"hard":[], "dense":[]}
    used_qids = []
    oracle_dense = []  # best achievable: predict p_mc -> error vs p_mc is 0; vs hard is (p_mc-y)^2
    for q in questions:
        key = (q.get("game_id", "?"), q["question_id"])
        if key not in p_mc or key not in usable:
            continue
        if q.get("question_type") and q["question_type"] != "binary":
            continue
        # Prefer the baseline-fork single future as the hard label (same config
        # as p_mc); fall back to the eval JSON ground_truth if unavailable.
        if key in base01:
            y01 = base01[key]
        else:
            y01 = 1.0 if q.get("ground_truth") in (True, 1, "true", "True") else 0.0
        pm = p_mc[key]
        used_qids.append(key)
        preds = q.get("predictions", {})
        for model, pr in preds.items():
            prob = pr.get("probability")
            if prob is None or pr.get("error"):
                continue
            per_model.setdefault(model, {"hard": [], "dense": []})
            per_model[model]["hard"].append((prob - y01) ** 2)
            per_model[model]["dense"].append((prob - pm) ** 2)

    rows = []
    for model, d in per_model.items():
        if not d["hard"]:
            continue
        bh = sum(d["hard"]) / len(d["hard"])
        bd = sum(d["dense"]) / len(d["dense"])
        rows.append({"model": model, "n": len(d["hard"]),
                     "brier_hard": bh, "brier_dense": bd})

    # Rank (lower brier = better = rank 1)
    by_hard = sorted(rows, key=lambda r: r["brier_hard"])
    by_dense = sorted(rows, key=lambda r: r["brier_dense"])
    rank_hard = {r["model"]: i + 1 for i, r in enumerate(by_hard)}
    rank_dense = {r["model"]: i + 1 for i, r in enumerate(by_dense)}
    for r in rows:
        r["rank_hard"] = rank_hard[r["model"]]
        r["rank_dense"] = rank_dense[r["model"]]
        r["rank_shift"] = r["rank_hard"] - r["rank_dense"]

    rho = spearman([r["brier_hard"] for r in rows], [r["brier_dense"] for r in rows])

    print(f"Questions used: {len(used_qids)} (>= {args.min_rollouts} rollouts), "
          f"models: {len(rows)}")
    print(f"Spearman(rank_hard, rank_dense) = {rho:.4f}")
    print()
    print(f"{'model':<48} {'n':>4} {'BrierHard':>10} {'BrierDense':>11} {'rH':>3} {'rD':>3} {'shift':>5}")
    for r in sorted(rows, key=lambda r: r["rank_hard"]):
        print(f"{r['model']:<48} {r['n']:>4} {r['brier_hard']:>10.4f} "
              f"{r['brier_dense']:>11.4f} {r['rank_hard']:>3} {r['rank_dense']:>3} "
              f"{r['rank_shift']:>+5}")

    out = {"spearman": rho, "n_questions": len(used_qids),
           "min_rollouts": args.min_rollouts, "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
