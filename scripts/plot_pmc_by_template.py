#!/usr/bin/env python3
"""Faceted p_mc distribution: one histogram panel per question template.

Usage:
    uv run python scripts/plot_pmc_by_template.py \\
        --mc tmp/mc_resolve/seed0_h1.json ... \\
        --questions-dir data/questions_mc \\
        --outfile docs/assets/mc_dense/5_pmc_by_template.png
"""
from __future__ import annotations

import json
import glob
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", nargs="+", required=True)
    ap.add_argument("--questions-dir", default="data/questions_mc")
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--min-rollouts", type=int, default=15)
    args = ap.parse_args()

    # game_id -> {question_id: template_id}
    qmap: dict[str, dict] = {}
    for qf in glob.glob(f"{args.questions_dir}/*/questions.json"):
        qb = json.loads(Path(qf).read_text())
        gid = qb.get("game_id", Path(qf).parent.name)
        qmap[gid] = {q["question_id"]: q["template_id"] for q in qb["questions"]}

    # template -> list of p_mc
    by_tmpl: dict[str, list] = defaultdict(list)
    for path in args.mc:
        d = json.loads(Path(path).read_text())
        gid = d.get("config", {}).get("game_id", "?")
        nr = d.get("n_rollouts", {})
        for qid, p in d.get("p_mc", {}).items():
            if nr.get(qid, 0) < args.min_rollouts:
                continue
            tmpl = qmap.get(gid, {}).get(qid)
            if tmpl:
                by_tmpl[tmpl].append(p)

    # Order panels by descending sample size
    templates = sorted(by_tmpl, key=lambda t: -len(by_tmpl[t]))
    n_total = sum(len(v) for v in by_tmpl.values())

    ncol = 4
    nrow = int(np.ceil(len(templates) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.5 * nrow),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    bins = np.linspace(0, 1, 11)

    for ax, tmpl in zip(axes, templates):
        vals = np.array(by_tmpl[tmpl])
        interior = np.mean((vals > 0.1) & (vals < 0.9)) if len(vals) else 0
        ax.hist(vals, bins=bins, color="#4C72B0", edgecolor="white")
        ax.axvline(0.5, color="grey", ls="--", lw=0.8)
        ax.set_title(f"{tmpl}\n(n={len(vals)}, {interior:.0%} in 0.1–0.9)",
                     fontsize=9)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=7)

    for ax in axes[len(templates):]:
        ax.axis("off")

    fig.suptitle(f"Dense 'real answer' (p_mc) distribution by question template "
                 f"(N={n_total}, 20 rollouts each)", fontsize=12, y=1.00)
    fig.supxlabel("p_mc = P(event) across rollouts", fontsize=10)
    fig.supylabel("number of questions", fontsize=10)
    fig.tight_layout()
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.outfile, dpi=130, bbox_inches="tight")
    print(f"Wrote {args.outfile}  ({len(templates)} templates, N={n_total})")
    for t in templates:
        v = np.array(by_tmpl[t])
        print(f"  {t:24} n={len(v):3}  mean={v.mean():.2f}  "
              f"interior={np.mean((v>0.1)&(v<0.9)):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
