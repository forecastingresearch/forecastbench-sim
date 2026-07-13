#!/usr/bin/env python3
"""On-pod eval of base model + verl checkpoints on the FB-Sim val set.

OpenForecaster eval protocol: n=3 samples, temperature 0.6, top-p 0.95, thinking on,
last <probability> tag after </think>. Scores every sample against BOTH targets
(p_mc dense and baseline 0/1 hard) regardless of training arm.

verl FSDP checkpoints are merged to HF format first via verl's model_merger.

Usage (on pod, forecast venv active):
  python /workspace/eval_checkpoints.py --model /workspace/models/Qwen3-8B \
      --tag base --val /workspace/data/val_dense.jsonl --out /workspace/evals/base.json
  python /workspace/eval_checkpoints.py --ckpt /workspace/ckpts/qwen3-8b-dense/global_step_10 \
      --tag dense-s10 --val /workspace/data/val_dense.jsonl --out /workspace/evals/dense_s10.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROB_RE = re.compile(r"<probability>(.*?)</probability>", re.DOTALL)


def parse_prob(text: str) -> float | None:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    matches = PROB_RE.findall(text)
    if not matches:
        return None
    try:
        v = float(matches[-1].strip())
    except ValueError:
        return None
    return min(1.0, max(0.0, v)) if -0.01 <= v <= 1.01 else None


def merge_ckpt(ckpt: str) -> str:
    target = str(Path(ckpt) / "merged_hf")
    if not (Path(target) / "config.json").exists():
        subprocess.run(
            [sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
             "--local_dir", str(Path(ckpt) / "actor"), "--target_dir", target],
            check=True)
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF model dir (base)")
    ap.add_argument("--ckpt", help="verl global_step dir (will be merged)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_dir = args.model or merge_ckpt(args.ckpt)
    rows = [json.loads(l) for l in open(args.val)]

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    prompts = [tok.apply_chat_template(r["prompt"], tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=True) for r in rows]
    llm = LLM(model=model_dir, tensor_parallel_size=1, max_model_len=17000,
              gpu_memory_utilization=0.85)
    sp = SamplingParams(n=args.n, temperature=0.6, top_p=0.95, max_tokens=8192)
    outs = llm.generate(prompts, sp)

    results, briers_d, briers_h, n_fail = [], [], [], 0
    for r, o in zip(rows, outs):
        info = r["extra_info"]
        preds = [parse_prob(c.text) for c in o.outputs]
        for p in preds:
            if p is None:
                n_fail += 1
            else:
                briers_d.append((p - info["p_mc"]) ** 2)
                briers_h.append((p - info["baseline_01"]) ** 2)
        results.append({"qid": info["qid"], "template_id": info["template_id"],
                        "p_mc": info["p_mc"], "hard": info["baseline_01"],
                        "preds": preds})
    n_tot = len(rows) * args.n
    summary = {"tag": args.tag, "model": model_dir, "n_samples": n_tot,
               "parse_rate": (n_tot - n_fail) / n_tot,
               "brier_dense": sum(briers_d) / len(briers_d) if briers_d else None,
               "brier_hard": sum(briers_h) / len(briers_h) if briers_h else None}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": results}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
