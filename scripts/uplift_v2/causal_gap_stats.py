#!/usr/bin/env python3
"""Confounding-gap certification stats for the scaled seed2 causal cells.

For each question with baseline N>=100 and do-arm N>=100:
  p0    = P(Y) baseline
  p_do  = P(Y | do X)
  p_obs = P(Y | X occurred naturally in baseline rollouts)   (n_obs varies)
  gap   = (p_obs - p0) - (p_do - p0) = p_obs - p_do

Tests, per question:
  z_do  : two-proportion z for p_do vs p0 (independent arms)
  z_gap : two-proportion z for p_obs vs p_do (obs subset vs do arm)
Reports counts above |z|>=1.96 and |z|>=3.29 (~Bonferroni for ~50 tests),
plus the top questions. Compares with binomial-null expectation.

Usage: uv run python scripts/uplift_v2/causal_gap_stats.py [seed]
"""
from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

CAUSAL = Path("tmp/causal")
CHOICES = json.loads((CAUSAL / "phase2_choices.json").read_text())


def rollout_answers(manifest: dict) -> dict[str, dict[str, bool]]:
    out: dict[str, dict[str, bool]] = {}
    for qid, recs in manifest["answers"].items():
        out[qid] = {r["tag"]: r["answer"] for r in recs if r["answer"] is not None}
    return out


def x_happened(rollout_path: Path, player: int, tech: str) -> bool:
    gd = json.load(gzip.open(rollout_path, "rt"))
    return any(e["type"] == "tech_discovered" and str(e.get("player_id")) == str(player)
               and e["description"].endswith(f"discovered {tech}") and 60 < e["turn"] <= 70
               for e in gd["events"])


def two_prop_z(p1: float, n1: int, p2: float, n2: int) -> float:
    pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(max(pool * (1 - pool) * (1 / n1 + 1 / n2), 1e-12))
    return (p1 - p2) / se


def main() -> int:
    seed = sys.argv[1] if len(sys.argv) > 1 else "2"
    choice = CHOICES[seed]
    base_dir = CAUSAL / f"seed{seed}" / "baseline"
    do_dirs = [d for d in (CAUSAL / f"seed{seed}").glob("do-*") if d.is_dir()]
    assert len(do_dirs) == 1
    base_m = json.loads((base_dir / "manifest.json").read_text())
    do_m = json.loads((do_dirs[0] / "manifest.json").read_text())

    x_tags = set()
    all_tags = set()
    for rp in sorted((base_dir / "rollouts").glob("*.json.gz")):
        tag = rp.name.split(".")[0]
        all_tags.add(tag)
        if x_happened(rp, choice["player"], choice["tech"]):
            x_tags.add(tag)

    base_a = rollout_answers(base_m)
    do_a = rollout_answers(do_m)

    rows = []
    for qid in sorted(base_a):
        if qid not in do_a:
            continue
        b = base_a[qid]
        d = do_a[qid]
        n0, n1 = len(b), len(d)
        if n0 < 100 or n1 < 100:
            continue
        p0 = sum(b.values()) / n0
        p_do = sum(d.values()) / n1
        obs = [v for t, v in b.items() if t in x_tags]
        n_obs = len(obs)
        p_obs = sum(obs) / n_obs if n_obs else None
        z_do = two_prop_z(p_do, n1, p0, n0)
        z_gap = (two_prop_z(p_obs, n_obs, p_do, n1)
                 if n_obs >= 5 and p_obs is not None else None)
        rows.append({"qid": qid, "n0": n0, "n1": n1, "n_obs": n_obs,
                     "p0": p0, "p_do": p_do, "p_obs": p_obs,
                     "delta_do": p_do - p0,
                     "gap": (p_obs - p_do) if p_obs is not None else None,
                     "z_do": z_do, "z_gap": z_gap})

    n_q = len(rows)
    print(f"seed{seed}: {n_q} questions, baseline N={rows[0]['n0']}, "
          f"do N={rows[0]['n1']}, X-observed n={rows[0]['n_obs']} "
          f"(X = player {choice['player']} discovers {choice['tech']} by t70)")

    for label, key in (("delta_do (z_do)", "z_do"), ("gap (z_gap)", "z_gap")):
        zs = [r[key] for r in rows if r[key] is not None]
        sig196 = sum(1 for z in zs if abs(z) >= 1.96)
        sig329 = sum(1 for z in zs if abs(z) >= 3.29)
        print(f"\n{label}: n={len(zs)}  |z|>=1.96: {sig196} "
              f"(chance ~{0.05*len(zs):.1f})  |z|>=3.29: {sig329} "
              f"(chance ~{0.001*len(zs):.2f})")
        top = sorted((r for r in rows if r[key] is not None),
                     key=lambda r: -abs(r[key]))[:8]
        for r in top:
            print(f"  {r['qid']} p0={r['p0']:.2f} p_do={r['p_do']:.2f} "
                  f"p_obs={('%.2f' % r['p_obs']) if r['p_obs'] is not None else '--'} "
                  f"n_obs={r['n_obs']} delta_do={r['delta_do']:+.2f} "
                  f"gap={('%+.2f' % r['gap']) if r['gap'] is not None else '--'} "
                  f"z_do={r['z_do']:+.2f} "
                  f"z_gap={('%+.2f' % r['z_gap']) if r['z_gap'] is not None else '--'}")

    out = CAUSAL / f"gap_stats_seed{seed}.json"
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
