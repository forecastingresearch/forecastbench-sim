#!/usr/bin/env python3
"""Build SFT + RL training JSONL from Freeciv questions + p_mc + world reports.

Joins:
  - data/questions_mc/{game}/questions.json       (questions)
  - data/questions_mc/{game}/world_report/turn_060_report.txt
  - tmp/mc_resolve/{game}_h1.json                 (p_mc, baseline, n_rollouts)

Produces:
  - train.jsonl, val.jsonl

Each record:
  {
    "game_id": "seed0",
    "qid": "q0000",
    "template_id": "tech_comparative",
    "prompt": "<system+report+question+instruction>",
    "p_mc": 0.55,
    "baseline_01": 1,
    "n_rollouts": 20,
    "weight": 4*p*(1-p),  # information weight
  }

Held-out by GAME, not question, so an unseen game tests generalization.

Usage:
    uv run python runpod/01_build_training_data.py \\
        --questions-dir data/questions_mc \\
        --mc-dir tmp/mc_resolve \\
        --val-games seed3 \\
        --output-dir data/training
"""
from __future__ import annotations

import json
import argparse
import glob
import random
from pathlib import Path

PROMPT_TEMPLATE = """You are a forecaster. Below is a partial report on a Freeciv game in progress, observed at turn 60. Five AI civilizations are competing. Use this report to estimate the probability of the event in the question.

============================ WORLD REPORT (turn 60) ============================
{report}
============================ END REPORT ============================

Question: {question}

Output your reasoning briefly, then end with a single line of exactly this form:
PROBABILITY: 0.xx

(where 0.xx is a number between 0 and 1, representing the probability the event resolves YES)."""


def build_record(game_id, qid, q, report, p_mc, baseline_01, n_rollouts) -> dict:
    prompt = PROMPT_TEMPLATE.format(report=report.strip(),
                                    question=q["question_text"].strip())
    return {
        "game_id": game_id,
        "qid": qid,
        "template_id": q.get("template_id"),
        "horizon": q.get("horizon"),
        "resolution_turn": q.get("resolution_turn"),
        "prompt": prompt,
        "p_mc": float(p_mc),
        "baseline_01": int(baseline_01) if baseline_01 is not None else None,
        "n_rollouts": int(n_rollouts),
        "weight": 4.0 * p_mc * (1.0 - p_mc),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-dir", default="data/questions_mc")
    ap.add_argument("--mc-dir", default="tmp/mc_resolve")
    ap.add_argument("--val-games", nargs="+", required=True,
                    help="Game IDs reserved for validation (held out from train).")
    ap.add_argument("--output-dir", default="data/training")
    ap.add_argument("--min-rollouts", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    train, val = [], []
    val_games = set(args.val_games)

    for mc_file in sorted(glob.glob(f"{args.mc_dir}/*_h1.json")):
        d = json.loads(Path(mc_file).read_text())
        game_id = d.get("config", {}).get("game_id")
        if not game_id:
            continue
        snapshot = d.get("config", {}).get("snapshot_turn", 60)

        qpath = Path(args.questions_dir) / game_id / "questions.json"
        rpath = Path(args.questions_dir) / game_id / "world_report" / f"turn_{snapshot:03d}_report.txt"
        if not qpath.exists() or not rpath.exists():
            print(f"[skip] {game_id}: missing questions or report")
            continue
        qb = json.loads(qpath.read_text())
        report = rpath.read_text()
        qmap = {q["question_id"]: q for q in qb["questions"]}
        baselines = d.get("baseline") or {}
        n_roll = d.get("n_rollouts", {})

        kept = 0
        for qid, p_mc in d.get("p_mc", {}).items():
            if n_roll.get(qid, 0) < args.min_rollouts:
                continue
            q = qmap.get(qid)
            if q is None:
                continue
            rec = build_record(game_id, qid, q, report, p_mc,
                               baselines.get(qid), n_roll.get(qid, 0))
            (val if game_id in val_games else train).append(rec)
            kept += 1
        print(f"[{game_id}] kept {kept}  (in val={game_id in val_games})")

    rng.shuffle(train)
    rng.shuffle(val)
    with open(out / "train.jsonl", "w") as f:
        for r in train: f.write(json.dumps(r) + "\n")
    with open(out / "val.jsonl", "w") as f:
        for r in val: f.write(json.dumps(r) + "\n")

    print(f"\nWrote {out}/train.jsonl ({len(train)} examples)")
    print(f"Wrote {out}/val.jsonl ({len(val)} examples)")
    if train:
        mean_p = sum(r["p_mc"] for r in train) / len(train)
        mean_w = sum(r["weight"] for r in train) / len(train)
        n_extreme = sum(1 for r in train if r["p_mc"] < 0.05 or r["p_mc"] > 0.95)
        print(f"  train mean p_mc={mean_p:.3f}, mean weight={mean_w:.3f}, "
              f"extreme(<0.05 or >0.95)={n_extreme}/{len(train)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
