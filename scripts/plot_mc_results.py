#!/usr/bin/env python3
"""Visualizations for the MC dense-resolution experiment.

Produces:
  1. p_mc distribution histogram (the "real answer" spread vs hard 0/1)
  2. p_mc vs baseline single-future label (how often one future disagrees
     with the MC frequency)
  3. per-model Brier: hard (vs 0/1) vs dense (vs p_mc), paired bars
  4. rank slope chart: rank under hard -> rank under dense

Usage:
    uv run python scripts/plot_mc_results.py \\
        --ranking tmp/mc_eval/ranking.json \\
        --mc tmp/mc_resolve/seed0_h1.json tmp/mc_resolve/seed1_h1.json ... \\
        --outdir tmp/mc_eval/plots
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def short(model: str) -> str:
    return model.split("/")[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", required=True)
    ap.add_argument("--mc", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-rollouts", type=int, default=15)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- gather per-question p_mc + baseline (keyed by (game, qid)) ----
    pmc_vals, base_vals, n_roll = [], [], []
    for path in args.mc:
        d = json.loads(Path(path).read_text())
        nr = d.get("n_rollouts", {})
        base = d.get("baseline") or {}
        for qid, p in d.get("p_mc", {}).items():
            if nr.get(qid, 0) < args.min_rollouts:
                continue
            pmc_vals.append(p)
            n_roll.append(nr.get(qid, 0))
            b = base.get(qid)
            base_vals.append(None if b is None else (1.0 if b else 0.0))
    pmc = np.array(pmc_vals)

    # ---- Plot 1: p_mc distribution ----
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(pmc, bins=np.linspace(0, 1, 21), color="#4C72B0", edgecolor="white")
    frac_interior = np.mean((pmc > 0.1) & (pmc < 0.9))
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlabel("MC resolution  p_mc  =  P(event) over 20 rollouts")
    ax.set_ylabel("number of questions")
    ax.set_title(f"Dense 'real answer' distribution (N={len(pmc)} questions)\n"
                 f"{frac_interior:.0%} land in (0.1, 0.9) — genuinely uncertain, "
                 f"not 0/1")
    fig.tight_layout()
    fig.savefig(outdir / "1_pmc_distribution.png", dpi=130)
    plt.close(fig)

    # ---- Plot 2: p_mc vs baseline single-future label ----
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    rng = np.random.default_rng(0)
    bv = np.array([b for b in base_vals], dtype=float)
    jitter = (rng.random(len(bv)) - 0.5) * 0.12
    disagree = np.mean(np.abs(pmc - bv) > 0.5)
    colors = ["#C44E52" if abs(p - b) > 0.5 else "#55A868"
              for p, b in zip(pmc, bv)]
    ax.scatter(bv + jitter, pmc, c=colors, alpha=0.6, s=22, edgecolor="none")
    ax.set_xticks([0, 1])
    ax.set_xlabel("baseline single-future label (one rollout) = 0 / 1")
    ax.set_ylabel("MC frequency  p_mc")
    ax.set_title(f"One future vs the ensemble\n"
                 f"red = single future disagrees with MC majority "
                 f"({disagree:.0%} of questions)")
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    fig.tight_layout()
    fig.savefig(outdir / "2_pmc_vs_baseline.png", dpi=130)
    plt.close(fig)

    # ---- ranking data ----
    R = json.loads(Path(args.ranking).read_text())
    rows = sorted(R["rows"], key=lambda r: r["rank_hard"])
    models = [short(r["model"]) for r in rows]
    bh = [r["brier_hard"] for r in rows]
    bd = [r["brier_dense"] for r in rows]

    # ---- Plot 3: paired Brier bars ----
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(models))
    ax.barh(y - 0.2, bh, height=0.4, color="#DD8452", label="Brier vs 0/1 (hard)")
    ax.barh(y + 0.2, bd, height=0.4, color="#4C72B0", label="Brier vs p_mc (dense)")
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Brier score (lower = better)")
    ax.set_title(f"Scoring against hard 0/1 vs dense MC probability\n"
                 f"Spearman(rank_hard, rank_dense) = {R['spearman']:.3f}, "
                 f"N={R['n_questions']} questions")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "3_brier_paired.png", dpi=130)
    plt.close(fig)

    # ---- Plot 4: rank slope chart ----
    fig, ax = plt.subplots(figsize=(6.2, 6.5))
    for r in rows:
        rh, rd = r["rank_hard"], r["rank_dense"]
        color = "#C44E52" if rh != rd else "#888888"
        ax.plot([0, 1], [rh, rd], "-o", color=color, lw=1.6, ms=5)
        ax.text(-0.04, rh, short(r["model"]), ha="right", va="center", fontsize=7.5)
        ax.text(1.04, rd, short(r["model"]), ha="left", va="center", fontsize=7.5)
    ax.set_xlim(-0.9, 1.9)
    ax.set_ylim(len(rows) + 0.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["rank by\nBrier vs 0/1", "rank by\nBrier vs p_mc"])
    ax.set_ylabel("rank (1 = best)")
    ax.set_title("Does dense scoring reorder models?\n"
                 "red = moved")
    ax.set_yticks(range(1, len(rows) + 1))
    fig.tight_layout()
    fig.savefig(outdir / "4_rank_slope.png", dpi=130)
    plt.close(fig)

    print(f"Wrote 4 plots to {outdir}/")
    print(f"  p_mc interior fraction (0.1-0.9): {frac_interior:.0%}")
    print(f"  single-future disagreement with MC: {disagree:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
