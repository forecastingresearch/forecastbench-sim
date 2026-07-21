#!/usr/bin/env python3
"""Convert training-record jsonl (build_training_jsonl.py output) into Fireworks
RFT (eval-protocol) dataset rows: {messages, ground_truth, metadata}.

Prompt rendering is byte-identical to build_of_dataset.py (OF binary prompt),
which is what the deployed evaluator (accounts/jaeholee/evaluators/test-forecast-
test-forecast) was trained/validated against in the calibration job.

Usage:
  uv run python scripts/uplift_v2/build_rft_dataset.py \
      --records data/training_v2/train.jsonl \
      --outdir data/uplift_v2/rft_v2 --prefix train
  -> writes {prefix}_dense.jsonl and {prefix}_hard.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_of_dataset import build_row  # noqa: E402  (reuses OF prompt + decompose)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.records)]
    order = list(range(len(recs)))
    random.Random(args.seed).shuffle(order)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for arm in ("dense", "hard"):
        rows = []
        for i in order:
            vr = build_row(recs[i], arm)  # verl-format row
            rows.append({
                "messages": [{"role": "user",
                              "content": vr["prompt"][0]["content"]}],
                "ground_truth": f"{vr['reward_model']['ground_truth']:.4f}",
                "metadata": {k: vr["extra_info"][k]
                             for k in ("qid", "game_id", "template_id", "arm")},
            })
        path = outdir / f"{args.prefix}_{arm}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        lens = [len(r["messages"][0]["content"]) for r in rows]
        print(f"{path}: {len(rows)} rows, chars mean={sum(lens)//len(lens)} "
              f"max={max(lens)} (~{max(lens)//3} tokens max)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
