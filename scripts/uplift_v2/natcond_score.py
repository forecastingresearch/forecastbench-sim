#!/usr/bin/env python3
"""Score natural-conditional elicitations: CUS + direction + selectivity.

Per (question, event) cell, with model forecasts p̂(Y) (mean over samples) and
p̂(Y|X), and ground truth p_y, p_yx (subset of n_x rollouts):

  Brier_cond(model) = mean over conditioning-subset outcomes of (p̂(Y|X)-y)^2
                    = (p̂(Y|X)-p_yx)^2 + p_yx(1-p_yx)   [expected form]
  Brier_cond(no-update): same with p̂(Y).
  CUS = 1 - sum_w Brier_cond(model) / sum_w Brier_cond(no-update)
        (w = inverse-variance weights from se_delta; pooled, not per-cell)
  Direction accuracy: cells with |delta| > 2*se_delta — sign(dhat) vs sign(delta)
  Selectivity = mean|dhat| on effect cells / mean|dhat| on placebo cells
  Noise floor: CUS of a perfect forecaster (p̂(Y|X)=p_yx, p̂(Y)=p_y) = upper bound.

Usage: uv run python scripts/uplift_v2/natcond_score.py tmp/natcond/elicit_seed2_*.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def load(path):
    d = json.load(open(path))
    cells = {(c["qid"], c["event_id"]): c for c in json.load(open(d["cells"]))}
    base, cond = defaultdict(list), defaultdict(list)
    for r in d["results"]:
        if r.get("p") is None:
            continue
        if r["stage"] == "base":
            base[r["qid"]].append(r["p"])
        else:
            cond[(r["qid"], r["event_id"])].append(r["p"])
    return d["tag"], cells, base, cond


def analyze(path):
    tag, cells, base, cond = load(path)
    rows = []
    for key, c in cells.items():
        qid, eid = key
        if qid not in base or key not in cond:
            continue
        pb = sum(base[qid]) / len(base[qid])
        pc = sum(cond[key]) / len(cond[key])
        rows.append(dict(c, p_base_hat=pb, p_cond_hat=pc, dhat=pc - pb))
    n = len(rows)
    if not n:
        print(f"{tag}: no scored cells"); return

    def brier_exp(p_hat, c):
        return (p_hat - c["p_yx"]) ** 2 + c["p_yx"] * (1 - c["p_yx"])

    wsum = mnum = mden = pnum = 0.0
    for r in rows:
        w = 1.0 / max(r["se_delta"] ** 2, 1e-4)
        wsum += w
        mnum += w * brier_exp(r["p_cond_hat"], r)
        mden += w * brier_exp(r["p_base_hat"], r)
        pnum += w * brier_exp(r["p_yx"], r)  # perfect forecaster
    cus = 1 - mnum / mden
    ceiling_gap = 1 - pnum / mden  # perfect model's CUS with same weights

    eff = [r for r in rows if r["cls"] == "effect"]
    pla = [r for r in rows if r["cls"] == "placebo"]
    sign_ok = sum(1 for r in eff if r["dhat"] * r["delta"] > 0)
    mean_d_eff = sum(abs(r["dhat"]) for r in eff) / max(len(eff), 1)
    mean_d_pla = sum(abs(r["dhat"]) for r in pla) / max(len(pla), 1)
    # split by event kind
    kinds = {}
    for kind in ("specific", "vague"):
        kr = [r for r in rows if r["event_kind"] == kind]
        if not kr:
            continue
        kn = kd = 0.0
        for r in kr:
            w = 1.0 / max(r["se_delta"] ** 2, 1e-4)
            kn += w * brier_exp(r["p_cond_hat"], r)
            kd += w * brier_exp(r["p_base_hat"], r)
        kinds[kind] = 1 - kn / kd

    print(f"\n===== {tag} ({n} cells: {len(eff)} effect / {len(pla)} placebo) =====")
    print(f"CUS (pooled, inv-var weighted): {cus:+.4f}   "
          f"[perfect-forecaster ceiling: {ceiling_gap:+.4f}]")
    for kind, v in kinds.items():
        print(f"  CUS on {kind} events: {v:+.4f}")
    print(f"direction accuracy on effect cells: {sign_ok}/{len(eff)} "
          f"= {sign_ok/max(len(eff),1):.2f} (chance 0.5)")
    print(f"selectivity: mean|update| effect={mean_d_eff:.4f} vs "
          f"placebo={mean_d_pla:.4f}  ratio={mean_d_eff/max(mean_d_pla,1e-6):.2f}")
    b_base = sum((r["p_base_hat"] - r["p_y"]) ** 2 for r in rows) / n
    print(f"(baseline forecast quality: mean (p̂(Y)-p_y)^2 = {b_base:.4f})")
    return dict(tag=tag, cus=cus, ceiling=ceiling_gap, kinds=kinds,
                direction=sign_ok / max(len(eff), 1),
                selectivity=mean_d_eff / max(mean_d_pla, 1e-6))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
