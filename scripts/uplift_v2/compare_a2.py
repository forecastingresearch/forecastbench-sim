#!/usr/bin/env python3
"""A2 three-way analysis: base vs hard-RL (s1,s2) vs dense-RL (s1,s2).

Pairs rows by (split, game_id, qid, sample). For each model and split reports
Brier-vs-p_mc, Brier-vs-hard, parse rate, prediction distribution. Then paired
bootstrap diffs vs base (question-level) and world-level aggregation (mean
per-world Brier, sign test across the 20 unseen worlds). Also a post-hoc
calibration check: affine-recalibrated Brier (isotonic-lite via 2-param
Platt-ish scaling is overkill; we report mean-shift-corrected Brier).

Usage: uv run python scripts/uplift_v2/compare_a2.py tmp/uplift_v2/a2_*.json
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict


def load(path):
    d = json.load(open(path))
    rows = {}
    for r in d["results"]:
        rows[(r["split"], r["game_id"], r["qid"], r["sample"])] = r
    return d["tag"], rows


def boot_ci(diffs, n=5000):
    rng = random.Random(0)
    ms = []
    for _ in range(n):
        s = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        ms.append(sum(s) / len(s))
    ms.sort()
    return ms[int(.025 * n)], ms[int(.975 * n)]


def main():
    models = dict(load(p) for p in sys.argv[1:])
    assert "base" in models
    common = set.intersection(*(set(r) for r in models.values()))
    print(f"models: {sorted(models)} | paired rows: {len(common)}")

    for split in ("val", "test"):
        keys = sorted(k for k in common if k[0] == split)
        print(f"\n================ {split} (n={len(keys)}) ================")
        stats = {}
        for tag, rows in sorted(models.items()):
            ps = [rows[k] for k in keys]
            parsed = [r for r in ps if r.get("p") is not None]
            bp = sum((r["p"] - r["p_mc"]) ** 2 for r in parsed) / len(parsed)
            bh = sum((r["p"] - r["hard"]) ** 2 for r in parsed) / len(parsed)
            mp = sum(r["p"] for r in parsed) / len(parsed)
            sd = (sum((r["p"] - mp) ** 2 for r in parsed) / len(parsed)) ** .5
            conf = sum(abs(r["p"] - .5) for r in parsed) / len(parsed)
            # mean-shift-corrected brier (removes pure bias miscalibration)
            bias = mp - sum(r["p_mc"] for r in parsed) / len(parsed)
            bc = sum((r["p"] - bias - r["p_mc"]) ** 2 for r in parsed) / len(parsed)
            stats[tag] = dict(brier_pmc=bp, brier_hard=bh, mean_p=mp, sd=sd,
                              conf=conf, brier_shift_corr=bc,
                              parse=len(parsed) / len(ps))
            print(f"{tag:10s} brier_pmc={bp:.4f} brier_hard={bh:.4f} "
                  f"shiftcorr={bc:.4f} mean_p={mp:.3f} sd={sd:.3f} "
                  f"|p-.5|={conf:.3f} parse={len(parsed)/len(ps):.3f}")

        print("\npaired diffs vs base (brier_pmc; negative = model better):")
        for tag in sorted(models):
            if tag == "base":
                continue
            diffs = []
            for k in keys:
                a, b = models["base"][k], models[tag][k]
                if a.get("p") is not None and b.get("p") is not None:
                    diffs.append((b["p"] - b["p_mc"]) ** 2 -
                                 (a["p"] - a["p_mc"]) ** 2)
            lo, hi = boot_ci(diffs)
            m = sum(diffs) / len(diffs)
            print(f"  {tag:10s} {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
                  f"  {'SIG' if hi < 0 or lo > 0 else ''}")

        # pooled arms
        print("\npooled arm diffs (dense = s1+s2 avg, hard = s1+s2 avg):")
        for arm in ("dense", "hard"):
            tags = [t for t in models if t.startswith(arm)]
            diffs = []
            for k in keys:
                a = models["base"][k]
                if a.get("p") is None:
                    continue
                bs = [models[t][k] for t in tags]
                bs = [b for b in bs if b.get("p") is not None]
                if not bs:
                    continue
                mb = sum((b["p"] - b["p_mc"]) ** 2 for b in bs) / len(bs)
                diffs.append(mb - (a["p"] - a["p_mc"]) ** 2)
            lo, hi = boot_ci(diffs)
            print(f"  {arm:6s} {sum(diffs)/len(diffs):+.4f}  "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}]"
                  f"  {'SIG' if hi < 0 or lo > 0 else ''}")
        # dense vs hard head-to-head
        diffs = []
        for k in keys:
            ds = [models[t][k] for t in models if t.startswith("dense")]
            hs = [models[t][k] for t in models if t.startswith("hard")]
            ds = [x for x in ds if x.get("p") is not None]
            hs = [x for x in hs if x.get("p") is not None]
            if ds and hs:
                md = sum((x["p"] - x["p_mc"]) ** 2 for x in ds) / len(ds)
                mh = sum((x["p"] - x["p_mc"]) ** 2 for x in hs) / len(hs)
                diffs.append(md - mh)
        lo, hi = boot_ci(diffs)
        print(f"  dense-vs-hard {sum(diffs)/len(diffs):+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  {'SIG' if hi < 0 or lo > 0 else ''}")

        # world-level: per-world mean brier, count worlds where model < base
        print("\nworld-level (per-world mean brier_pmc, wins vs base):")
        for tag in sorted(models):
            if tag == "base":
                continue
            wins, tot = 0, 0
            for gid in sorted({k[1] for k in keys}):
                gk = [k for k in keys if k[1] == gid]
                a = [models["base"][k] for k in gk]
                b = [models[tag][k] for k in gk]
                a = [(r["p"] - r["p_mc"]) ** 2 for r in a if r.get("p") is not None]
                b = [(r["p"] - r["p_mc"]) ** 2 for r in b if r.get("p") is not None]
                if a and b:
                    tot += 1
                    if sum(b) / len(b) < sum(a) / len(a):
                        wins += 1
            print(f"  {tag:10s} beats base in {wins}/{tot} worlds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
