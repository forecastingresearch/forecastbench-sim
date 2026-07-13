#!/usr/bin/env python3
"""Build OpenForecaster-format verl datasets from FB-Sim MC-labeled records.

Renders each record with OpenForecaster's binary forecasting prompt (verbatim from
their prompt_utils.py, no-retrieval variant), so the RL replication changes ONLY the
data source. Two arms, byte-identical except reward_model.ground_truth:

  hard  : ground_truth = baseline_01 (the single canonical rollout's 0/1)
  dense : ground_truth = p_mc        (MC event frequency over ~20 rollouts)

The verifier's binary branch is generalized (2-line patch, see of_patch/) to
reward = -(p - ground_truth)^2, which reproduces OpenForecaster's exact hard-label
reward when ground_truth is 0/1.

Usage:
  uv run python scripts/uplift_v2/build_of_dataset.py \
      --records data/training/train.jsonl --split train --outdir data/uplift_v2
  uv run python scripts/uplift_v2/build_of_dataset.py \
      --records data/training/val.jsonl --split val --outdir data/uplift_v2
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# OpenForecaster binary prompt, no-retrieval variant — verbatim skeleton.
OF_BINARY_PROMPT = """You will be asked a binary forecasting question. You have to come up with the best estimate for whether the event asked in the question happens or happened. Please provide your reasoning before stating how likely is the event asked in the question to happen (your confidence of it resolving YES).

Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}

Think step by step about the information provided, reason about uncertainty and put your final confidence for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. For example, if p = 0.6, and the event resolves to NO, then your score will be (- (0.6^2)) = -0.36 whereas if the event resolves to YES, then your score would be - (1 - 0.6)^2 = -0.16. Thus, the range of the score is [-1, 0]. If you output probability more than 0.5, then it is assumed that you think the event will likely resolve to "YES" while if you output probability less than 0.5, then it is assumed that you think the event will likely resolve to "NO". YOU HAVE TO MAXIMIZE YOUR BRIER SCORE.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags."""

REPORT_RE = re.compile(
    r"={10,} WORLD REPORT \(turn \d+\) ={10,}\n(.*?)\n={10,} END REPORT ={10,}",
    re.DOTALL)
QUESTION_RE = re.compile(r"\nQuestion: (.+?)\n")


def decompose(prompt: str) -> tuple[str, str]:
    rep = REPORT_RE.search(prompt)
    q = QUESTION_RE.search(prompt, rep.end() if rep else 0)
    if not rep or not q:
        raise ValueError("prompt did not decompose")
    return rep.group(1).strip(), q.group(1).strip()


def build_row(rec: dict, arm: str) -> dict:
    report, question = decompose(rec["prompt"])
    background = (
        "This is a partial report on a FreeCiv game simulation in progress, observed "
        "at turn 60. Five AI civilizations are competing.\n\n" + report)
    resolution_criteria = (
        f"Resolves YES if the answer to the question is affirmative in the simulation "
        f"state at turn {rec['resolution_turn']}, as determined by the game's "
        f"recorded metrics.")
    prompt = OF_BINARY_PROMPT.format(
        question_title=question, background=background,
        resolution_criteria=resolution_criteria)
    target = float(rec["p_mc"]) if arm == "dense" else float(int(rec["baseline_01"]))
    return {
        "data_source": "binary/fbsim",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "forecasting",
        "reward_model": {"style": "rule", "ground_truth": target},
        "extra_info": {
            # verifier.py reads these two: question_source routes to the binary
            # reward branch; resolution is the reward target.
            "question_source": "binary/fbsim",
            "resolution": target,
            "question": question,
            "qid": rec["qid"], "game_id": rec["game_id"],
            "template_id": rec["template_id"], "horizon": rec["horizon"],
            "resolution_turn": rec["resolution_turn"],
            "p_mc": float(rec["p_mc"]), "baseline_01": int(rec["baseline_01"]),
            "arm": arm,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=17, help="shuffle seed (same for both arms)")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.records)]
    order = list(range(len(recs)))
    random.Random(args.seed).shuffle(order)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for arm in ("hard", "dense"):
        rows = [build_row(recs[i], arm) for i in order]
        path = outdir / f"{args.split}_{arm}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        lens = [len(r["prompt"][0]["content"]) for r in rows]
        stats[arm] = {"n": len(rows), "path": str(path),
                      "chars_mean": sum(lens) // len(lens), "chars_max": max(lens)}
        # parquet for stock verl
        try:
            import pandas as pd
            pd.DataFrame(rows).to_parquet(path.with_suffix(".parquet"))
        except ImportError:
            pass
    # sanity: arms differ only in ground_truth/arm fields
    h = [json.loads(l) for l in open(outdir / f"{args.split}_hard.jsonl")]
    d = [json.loads(l) for l in open(outdir / f"{args.split}_dense.jsonl")]
    assert all(a["prompt"] == b["prompt"] and a["extra_info"]["qid"] == b["extra_info"]["qid"]
               for a, b in zip(h, d))
    print(json.dumps(stats, indent=1))
    print(f"approx max prompt tokens: ~{max(s['chars_max'] for s in stats.values()) // 4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
