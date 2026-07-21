#!/usr/bin/env python3
"""Gate-A1 decision: paired comparison of base vs RFT model eval outputs.

Pairs by (split, game_id, qid, sample). Reports per split:
  - mean reward (training convention: unparsed -> 0)
  - Brier vs p_mc and vs hard label (parsed-only, paired subset)
  - paired bootstrap CI on the difference (RFT - base)
  - prediction distribution shift (mean/SD), calibration-shrinkage check:
    Brier decomposition proxy — does RFT just predict closer to 0.5?

Usage:
  uv run python scripts/uplift_v2/compare_a1.py \
      tmp/uplift_v2/a1_base.json tmp/uplift_v2/a1_densecal.json
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict


def load(path: str) -> dict:
    d = json.load(open(path))
    rows = {}
    for r in d["results"]:
        key = (r["split"], r["game_id"], r["qid"], r["sample"])
        rows[key] = r
    return {"tag": d["tag"], "rows": rows}


def boot_ci(diffs: list[float], n: int = 5000) -> tuple[float, float]:
    rng = random.Random(0)
    means = []
    for _ in range(n):
        s = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(sum(s) / len(s))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> int:
    a = load(sys.argv[1])
    b = load(sys.argv[2])
    common = sorted(set(a["rows"]) & set(b["rows"]))
    print(f"paired rows: {len(common)} "
          f"(a-only {len(a['rows']) - len(common)}, b-only {len(b['rows']) - len(common)})")

    for split in ("train", "val"):
        keys = [k for k in common if k[0] == split]
        if not keys:
            continue
        print(f"\n=== {split} (n={len(keys)}) ===")

        def reward(r):
            return 1.0 - (r["p"] - r["p_mc"]) ** 2 if r.get("p") is not None else 0.0

        ra = [reward(a["rows"][k]) for k in keys]
        rb = [reward(b["rows"][k]) for k in keys]
        diffs = [y - x for x, y in zip(ra, rb)]
        lo, hi = boot_ci(diffs)
        print(f"reward  {a['tag']}={sum(ra)/len(ra):.4f}  {b['tag']}={sum(rb)/len(rb):.4f}  "
              f"diff={sum(diffs)/len(diffs):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

        # parsed-only paired Brier vs p_mc and vs hard
        pk = [k for k in keys if a["rows"][k].get("p") is not None
              and b["rows"][k].get("p") is not None]
        for target, label in (("p_mc", "brier_pmc"), ("hard", "brier_hard")):
            tk = [k for k in pk if a["rows"][k].get(target) is not None]
            if not tk:
                continue
            ba = [(a["rows"][k]["p"] - a["rows"][k][target]) ** 2 for k in tk]
            bb = [(b["rows"][k]["p"] - b["rows"][k][target]) ** 2 for k in tk]
            diffs = [y - x for x, y in zip(ba, bb)]
            lo, hi = boot_ci(diffs)
            print(f"{label}  {a['tag']}={sum(ba)/len(ba):.4f}  {b['tag']}={sum(bb)/len(bb):.4f}  "
                  f"diff={sum(diffs)/len(diffs):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] "
                  f"(neg = {b['tag']} better; n={len(tk)})")

        # shrinkage check
        for m in (a, b):
            ps = [m["rows"][k]["p"] for k in pk]
            mean_p = sum(ps) / len(ps)
            sd = (sum((p - mean_p) ** 2 for p in ps) / len(ps)) ** 0.5
            d05 = sum(abs(p - 0.5) for p in ps) / len(ps)
            print(f"  dist[{m['tag']}]: mean={mean_p:.3f} sd={sd:.3f} "
                  f"mean|p-0.5|={d05:.3f}")

        # per-template paired diff (reward)
        by_tpl = defaultdict(list)
        for k in keys:
            by_tpl[a["rows"][k]["template_id"]].append(
                reward(b["rows"][k]) - reward(a["rows"][k]))
        print("  per-template reward diff (RFT - base):")
        for t, ds in sorted(by_tpl.items(), key=lambda kv: -abs(sum(kv[1]) / len(kv[1]))):
            print(f"    {t:28s} {sum(ds)/len(ds):+.4f} (n={len(ds)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
