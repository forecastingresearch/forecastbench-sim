#!/usr/bin/env python3
"""Final A2 assembly + analysis. Merges the clean eval sources per model:
  base:     val from a2_base.json,     test from a2_base-test.json
  dense-s1: val from a2_dense-s1.json, test from a2_dense-s1-test.json
  dense-s2 / hard-s1 / hard-s2: both splits from their (clean) rerun files.
Reports per model: raw + affine-calibrated (fit val, apply test) Brier vs p_mc,
corr, distribution; paired bootstrap arm comparisons; world-level wins.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict

SRC = {
    "base": [("tmp/uplift_v2/a2_base.json", "val"),
             ("tmp/uplift_v2/a2_base-test.json", "test")],
    "dense-s1": [("tmp/uplift_v2/a2_dense-s1.json", "val"),
                 ("tmp/uplift_v2/a2_dense-s1-test.json", "test")],
    "dense-s2": [("tmp/uplift_v2/a2_dense-s2.json", "val"),
                 ("tmp/uplift_v2/a2_dense-s2.json", "test")],
    "hard-s1": [("tmp/uplift_v2/a2_hard-s1.json", "val"),
                ("tmp/uplift_v2/a2_hard-s1.json", "test")],
    "hard-s2": [("tmp/uplift_v2/a2_hard-s2.json", "val"),
                ("tmp/uplift_v2/a2_hard-s2.json", "test")],
}


def load_model(tag):
    rows = {}
    for path, split in SRC[tag]:
        d = json.load(open(path))
        for r in d["results"]:
            if r["split"] == split and r.get("p") is not None:
                rows[(split, r["game_id"], r["qid"], r["sample"])] = r
    return rows


def affine_fit(rows):
    v = [r for k, r in rows.items() if k[0] == "val"]
    n = len(v); sx = sum(r["p"] for r in v); sy = sum(r["p_mc"] for r in v)
    sxx = sum(r["p"] ** 2 for r in v); sxy = sum(r["p"] * r["p_mc"] for r in v)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    return a, b


def corr(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    c = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
    return c / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


def boot(diffs, n=5000):
    rng = random.Random(0); ms = []
    for _ in range(n):
        s = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        ms.append(sum(s) / len(s))
    ms.sort()
    return sum(diffs) / len(diffs), ms[int(.025 * n)], ms[int(.975 * n)]


def main():
    models = {t: load_model(t) for t in SRC}
    print("rows per model:", {t: len(r) for t, r in models.items()})
    cal = {t: affine_fit(r) for t, r in models.items()}

    # constant-0.5 reference per split
    qref = {}
    for k, r in models["base"].items():
        qref.setdefault((k[0], k[1], k[2]), r["p_mc"])
    for split in ("val", "test"):
        v = [p for (s, _, _), p in qref.items() if s == split]
        print(f"const-0.5 {split}: {sum((0.5-p)**2 for p in v)/len(v):.4f} "
              f"(n={len(v)})")

    print(f"\n{'model':10s} {'val_raw':>8s} {'test_raw':>9s} {'test_cal':>9s} "
          f"{'corr_t':>7s} {'|p-.5|':>7s} {'a':>6s} {'b':>6s}")
    for t, rows in models.items():
        a, b = cal[t]
        for split in ("test",):
            rs = [r for k, r in rows.items() if k[0] == split]
            vs = [r for k, r in rows.items() if k[0] == "val"]
            vraw = sum((r["p"] - r["p_mc"]) ** 2 for r in vs) / len(vs)
            raw = sum((r["p"] - r["p_mc"]) ** 2 for r in rs) / len(rs)
            c = sum((min(1, max(0, a + b * r["p"])) - r["p_mc"]) ** 2
                    for r in rs) / len(rs)
            cr = corr([r["p"] for r in rs], [r["p_mc"] for r in rs])
            conf = sum(abs(r["p"] - .5) for r in rs) / len(rs)
            print(f"{t:10s} {vraw:8.4f} {raw:9.4f} {c:9.4f} {cr:7.3f} "
                  f"{conf:7.3f} {a:6.3f} {b:6.3f}")

    # paired comparisons on test (question-level, sample-paired where possible)
    print("\npaired test diffs (Brier vs p_mc, negative = first better):")
    def paired(t1, t2):
        r1, r2 = models[t1], models[t2]
        ks = [k for k in r1 if k[0] == "test" and k in r2]
        if len(ks) < 100:  # sample indices may differ; pair at question level
            q1 = defaultdict(list); q2 = defaultdict(list)
            for k, r in r1.items():
                if k[0] == "test": q1[(k[1], k[2])].append(r)
            for k, r in r2.items():
                if k[0] == "test": q2[(k[1], k[2])].append(r)
            qs = set(q1) & set(q2)
            diffs = []
            for q in qs:
                m1 = sum((r["p"] - r["p_mc"]) ** 2 for r in q1[q]) / len(q1[q])
                m2 = sum((r["p"] - r["p_mc"]) ** 2 for r in q2[q]) / len(q2[q])
                diffs.append(m1 - m2)
        else:
            diffs = [(r1[k]["p"] - r1[k]["p_mc"]) ** 2 -
                     (r2[k]["p"] - r2[k]["p_mc"]) ** 2 for k in ks]
        m, lo, hi = boot(diffs)
        sig = "SIG" if hi < 0 or lo > 0 else "   "
        print(f"  {t1:>9s} vs {t2:9s}: {m:+.4f} [{lo:+.4f},{hi:+.4f}] {sig} "
              f"(n={len(diffs)})")
    for t in ("dense-s1", "dense-s2", "hard-s1", "hard-s2"):
        paired(t, "base")
    # pooled arms
    def pool_q(tags):
        out = defaultdict(list)
        for t in tags:
            for k, r in models[t].items():
                if k[0] == "test":
                    out[(k[1], k[2])].append((r["p"] - r["p_mc"]) ** 2)
        return {q: sum(v) / len(v) for q, v in out.items()}
    d = pool_q(["dense-s1", "dense-s2"]); h = pool_q(["hard-s1", "hard-s2"])
    qs = set(d) & set(h)
    m, lo, hi = boot([d[q] - h[q] for q in qs])
    print(f"  dense-arm vs hard-arm: {m:+.4f} [{lo:+.4f},{hi:+.4f}] "
          f"{'SIG' if hi < 0 or lo > 0 else ''} (n={len(qs)})")

    # world wins vs base
    print("\ntest world wins vs base (15 worlds):")
    bq = pool_q(["base"])
    for t in ("dense-s1", "dense-s2", "hard-s1", "hard-s2"):
        tq = pool_q([t])
        byw_t = defaultdict(list); byw_b = defaultdict(list)
        for (g, q), v in tq.items(): byw_t[g].append(v)
        for (g, q), v in bq.items(): byw_b[g].append(v)
        wins = sum(1 for g in byw_t
                   if sum(byw_t[g])/len(byw_t[g]) < sum(byw_b[g])/len(byw_b[g]))
        print(f"  {t}: {wins}/{len(byw_t)}")


if __name__ == "__main__":
    main()
