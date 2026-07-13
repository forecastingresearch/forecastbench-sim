#!/usr/bin/env python3
"""Phase-3 analysis for the causal golden-question pilot.

For each seed and question Y computes:
  P(Y)          — baseline-arm p_mc (N=40)
  P(Y|do X)     — intervention-arm p_mc (N=40)
  P(Y|X obs)    — baseline rollouts where X happened naturally (X = target player
                  discovered the chosen tech by turn 70)
  Delta_do  = P(Y|do X) - P(Y)
  Delta_obs = P(Y|X obs) - P(Y)
  gap       = Delta_obs - Delta_do   (confounding gap)

Reports the phase-4 gate (# questions with |Delta_do| > 0.15) and writes
tmp/causal/analysis.json.

Usage: uv run python scripts/uplift_v2/causal_analyze.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

CAUSAL = Path("tmp/causal")
CHOICES = json.loads((CAUSAL / "phase2_choices.json").read_text())


def rollout_answers(manifest: dict) -> dict[str, dict[str, bool]]:
    """answers keyed by qid -> {rollout_tag: bool}"""
    out: dict[str, dict[str, bool]] = {}
    for qid, recs in manifest["answers"].items():
        out[qid] = {r["tag"]: r["answer"] for r in recs if r["answer"] is not None}
    return out


def x_happened(rollout_path: Path, player: int, tech: str) -> bool:
    gd = json.load(gzip.open(rollout_path, "rt"))
    return any(e["type"] == "tech_discovered" and str(e.get("player_id")) == str(player)
               and e["description"].endswith(f"discovered {tech}") and 60 < e["turn"] <= 70
               for e in gd["events"])


def main() -> int:
    all_rows = []
    for seed, choice in sorted(CHOICES.items()):
        base_dir = CAUSAL / f"seed{seed}" / "baseline"
        do_dirs = [d for d in (CAUSAL / f"seed{seed}").glob("do-*") if d.is_dir()]
        assert len(do_dirs) == 1, do_dirs
        base_m = json.loads((base_dir / "manifest.json").read_text())
        do_m = json.loads((do_dirs[0] / "manifest.json").read_text())

        # conditioning set: baseline rollouts where X occurred naturally
        x_tags = set()
        for rp in sorted((base_dir / "rollouts").glob("*.json.gz")):
            if x_happened(rp, choice["player"], choice["tech"]):
                x_tags.add(rp.name.split(".")[0])

        base_a = rollout_answers(base_m)
        do_a = rollout_answers(do_m)
        n_x = len(x_tags)
        for qid in sorted(base_a):
            if qid not in do_a:
                continue
            b = base_a[qid]
            p_y = sum(b.values()) / len(b)
            d = do_a[qid]
            p_do = sum(d.values()) / len(d)
            obs = [v for t, v in b.items() if t in x_tags]
            p_obs = sum(obs) / len(obs) if obs else None
            row = {"seed": int(seed), "qid": qid, "x": f"p{choice['player']} {choice['tech']}",
                   "n_base": len(b), "n_do": len(d), "n_obs": len(obs),
                   "p_y": p_y, "p_do": p_do, "p_obs": p_obs,
                   "delta_do": p_do - p_y,
                   "delta_obs": (p_obs - p_y) if p_obs is not None else None,
                   "gap": (p_obs - p_do) if p_obs is not None else None}
            all_rows.append(row)
        print(f"seed{seed}: X = p{choice['player']} {choice['tech']} "
              f"(natural {n_x}/40), {len(base_a)} questions")

    big = [r for r in all_rows if abs(r["delta_do"]) > 0.15]
    big_gap = [r for r in all_rows if r["gap"] is not None and abs(r["gap"]) > 0.15]
    mean_abs_do = sum(abs(r["delta_do"]) for r in all_rows) / len(all_rows)
    summary = {
        "n_questions": len(all_rows),
        "mean_abs_delta_do": round(mean_abs_do, 4),
        "n_delta_do_gt_.15": len(big),
        "n_gap_gt_.15": len(big_gap),
        "gate_phase4": len(big) >= 10,
    }
    (CAUSAL / "analysis.json").write_text(json.dumps(
        {"summary": summary, "choices": CHOICES, "rows": all_rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print("\nTop |delta_do| questions:")
    for r in sorted(all_rows, key=lambda r: -abs(r["delta_do"]))[:8]:
        print(f"  seed{r['seed']} {r['qid']} do:{r['delta_do']:+.2f} "
              f"obs:{(r['delta_obs'] if r['delta_obs'] is not None else float('nan')):+.2f} "
              f"gap:{(r['gap'] if r['gap'] is not None else float('nan')):+.2f} ({r['x']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
