#!/usr/bin/env python3
"""Evaluate a checkpoint on (a) ForecastBench real-world questions and/or
(b) the Freeciv held-out val.jsonl. Computes Brier, fraction in extreme
bands, and (where applicable) base-rate baseline.

Uses vLLM for fast generation; loads LoRA adapter on top of base model.

Run on RunPod:
    python runpod/04_eval.py \\
        --base Qwen/Qwen3-14B --adapter runs/sft/adapter \\
        --forecastbench data/forecastbench/eval_post2025.csv \\
        --freeciv-val data/training/val.jsonl \\
        --output runs/sft/eval.json
"""
from __future__ import annotations

import re
import csv
import json
import argparse
from pathlib import Path

PROB_RE = re.compile(r"PROBABILITY:\s*([01](?:\.\d+)?|\.\d+)", re.IGNORECASE)


def parse_probability(text: str) -> float | None:
    m = PROB_RE.findall(text or "")
    if not m:
        return None
    try:
        p = float(m[-1])
    except ValueError:
        return None
    return p if 0.0 <= p <= 1.0 else None


FB_PROMPT = """You are a forecaster. You are estimating the probability of a real-world event resolving YES, as of the freeze date below. Do NOT assume knowledge of anything after the freeze date.

Freeze date: {freeze_date}
Source: {source}

Question: {question}

Background: {background}

Resolution criteria: {criteria}

Output a brief one-paragraph rationale, then end with a single line of exactly this form:
PROBABILITY: 0.xx

(where 0.xx is your probability in [0, 1] that the question resolves YES)."""


def build_fb_prompts(csv_path: str) -> list[tuple[str, str, float, dict]]:
    out = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            prompt = FB_PROMPT.format(
                freeze_date=r["freeze_date"], source=r["source"],
                question=r["question"],
                background=(r["background"] or "(none)"),
                criteria=(r["resolution_criteria"] or "(none)"))
            out.append((r["qid"], prompt, float(r["resolved_to"]),
                        {"source": r["source"],
                         "freeze_market_value": r["freeze_market_value"],
                         "freeze_date": r["freeze_date"]}))
    return out


def build_freeciv_prompts(jsonl_path: str):
    out = []
    for line in open(jsonl_path):
        r = json.loads(line)
        out.append((r["qid"], r["prompt"], float(r["p_mc"]),
                    {"game_id": r["game_id"], "template_id": r["template_id"],
                     "baseline_01": r.get("baseline_01"),
                     "n_rollouts": r.get("n_rollouts")}))
    return out


def brier_stats(rows: list[dict]) -> dict:
    n = len(rows)
    parsed = [r for r in rows if r["pred"] is not None]
    n_parsed = len(parsed)
    brier = sum((r["pred"] - r["target"]) ** 2 for r in parsed) / max(1, n_parsed)
    extreme = sum(1 for r in parsed if r["pred"] < 0.1 or r["pred"] > 0.9)
    base_rate = sum(r["target"] for r in parsed) / max(1, n_parsed)
    brier_baseline = sum((base_rate - r["target"]) ** 2 for r in parsed) / max(1, n_parsed)
    return {"n": n, "n_parsed": n_parsed, "brier": brier,
            "brier_constant_baseline": brier_baseline, "base_rate": base_rate,
            "frac_extreme_0.1_0.9": extreme / max(1, n_parsed)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-14B")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter path; omit to eval base model.")
    ap.add_argument("--forecastbench", default=None)
    ap.add_argument("--freeciv-val", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy. For variance-style eval bump to 0.7.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Subsample for quick checks.")
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    enable_lora = args.adapter is not None
    llm = LLM(model=args.base, enable_lora=enable_lora, max_lora_rank=64,
              gpu_memory_utilization=0.88, dtype="bfloat16",
              max_model_len=10240, enforce_eager=False)
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens,
                        stop=["</s>", "<|endoftext|>", "<|im_end|>"])
    lora_req = LoRARequest("adapter", 1, args.adapter) if enable_lora else None

    output = {"base": args.base, "adapter": args.adapter, "results": {}}

    suites = []
    if args.forecastbench:
        suites.append(("forecastbench", build_fb_prompts(args.forecastbench)))
    if args.freeciv_val:
        suites.append(("freeciv_val", build_freeciv_prompts(args.freeciv_val)))

    for name, items in suites:
        if args.limit:
            items = items[: args.limit]
        print(f"\n[{name}] generating on {len(items)} items...")
        prompts = [it[1] for it in items]
        outs = llm.generate(prompts, sp, lora_request=lora_req)
        rows = []
        for (qid, _prompt, target, meta), out in zip(items, outs):
            text = out.outputs[0].text if out.outputs else ""
            pred = parse_probability(text)
            rows.append({"qid": qid, "target": target, "pred": pred,
                         "meta": meta, "raw_tail": text[-300:]})
        stats = brier_stats(rows)
        print(f"  n={stats['n']} parsed={stats['n_parsed']} "
              f"Brier={stats['brier']:.4f} "
              f"(constant-baseline {stats['brier_constant_baseline']:.4f}, "
              f"base_rate={stats['base_rate']:.3f}) "
              f"extreme={stats['frac_extreme_0.1_0.9']:.0%}")
        output["results"][name] = {"stats": stats, "rows": rows}

    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
