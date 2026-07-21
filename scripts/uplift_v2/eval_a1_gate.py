#!/usr/bin/env python3
"""Gate A1: does the Fireworks RFT model (fbsim-rft-dense-cal) beat untrained
qwen3-4b on its own training data / seed-3 val?

Decoding matches RFT training exactly: temp 1.0, top_p 1.0, max_tokens 8192,
same OF binary prompt, parse last <probability> tag after </think>.

Row sources (normalized to {prompt, p_mc, hard, qid, template_id, game_id, split}):
  train: scripts/uplift_v2/fireworks_rft/development/forecast_dense.jsonl
         (+ forecast_hard.jsonl for the hard label, matched by qid+game_id)
  val:   data/uplift_v2/val_dense.jsonl (verl format)

Usage:
  set -a; source .env; set +a
  uv run python scripts/uplift_v2/eval_a1_gate.py \
      --model 'accounts/fireworks/models/qwen3-4b#accounts/jaeholee/deployments/cb-a1-qwen3-4b' \
      --tag base --samples-val 4 \
      --output tmp/uplift_v2/a1_base.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

API_BASE = "https://api.fireworks.ai/inference/v1"
PROB_RE = re.compile(r"<probability>\s*([0-9.eE+-]+)\s*</probability>")
ROOT = Path(__file__).resolve().parents[2]


def load_rows() -> list[dict]:
    rows = []
    dense_p = ROOT / "scripts/uplift_v2/fireworks_rft/development/forecast_dense.jsonl"
    hard_p = ROOT / "scripts/uplift_v2/fireworks_rft/development/forecast_hard.jsonl"
    hard_by_key = {}
    for line in open(hard_p):
        r = json.loads(line)
        hard_by_key[(r["metadata"]["game_id"], r["metadata"]["qid"])] = float(r["ground_truth"])
    for line in open(dense_p):
        r = json.loads(line)
        md = r["metadata"]
        rows.append({
            "prompt": r["messages"][0]["content"],
            "p_mc": float(r["ground_truth"]),
            "hard": hard_by_key.get((md["game_id"], md["qid"])),
            "qid": md["qid"], "template_id": md["template_id"],
            "game_id": md["game_id"], "split": "train",
        })
    for line in open(ROOT / "data/uplift_v2/val_dense.jsonl"):
        r = json.loads(line)
        ei = r["extra_info"]
        rows.append({
            "prompt": r["prompt"][0]["content"],
            "p_mc": float(ei["p_mc"]), "hard": float(ei["baseline_01"]),
            "qid": ei["qid"], "template_id": ei["template_id"],
            "game_id": ei["game_id"], "split": "val",
        })
    return rows


def call_chat(model: str, prompt: str, timeout: int = 600) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    req = urllib.request.Request(
        f"{API_BASE}/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}",
                 "Content-Type": "application/json",
                 "User-Agent": "civbench-a1-gate/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    ch = d["choices"][0]
    return {"text": ch["message"]["content"] or "",
            "finish": ch.get("finish_reason"),
            "usage": d.get("usage", {})}


def parse_prob(text: str) -> float | None:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    m = PROB_RE.findall(text)
    if not m:
        return None
    try:
        p = float(m[-1])
    except ValueError:
        return None
    if -0.01 <= p <= 1.01:
        return min(1.0, max(0.0, p))
    return None


def eval_one(model: str, row: dict, sample_idx: int, max_retries: int = 3) -> dict:
    last_err = None
    for attempt in range(max_retries):
        try:
            out = call_chat(model, row["prompt"])
            p = parse_prob(out["text"])
            return {
                "qid": row["qid"], "game_id": row["game_id"],
                "template_id": row["template_id"], "split": row["split"],
                "sample": sample_idx, "p": p, "p_mc": row["p_mc"],
                "hard": row["hard"], "finish": out["finish"],
                "completion_tokens": out["usage"].get("completion_tokens"),
            }
        except Exception as e:  # noqa: BLE001
            last_err = str(e)[:200]
            time.sleep(5 * (attempt + 1))
    return {"qid": row["qid"], "game_id": row["game_id"],
            "template_id": row["template_id"], "split": row["split"],
            "sample": sample_idx, "p": None, "p_mc": row["p_mc"],
            "hard": row["hard"], "error": last_err}


def summarize(results: list[dict]) -> dict:
    out = {}
    for split in ("train", "val"):
        rs = [r for r in results if r["split"] == split]
        parsed = [r for r in rs if r.get("p") is not None]
        if not rs:
            continue
        n = len(rs)
        def mean(xs):
            xs = list(xs)
            return sum(xs) / len(xs) if xs else None
        # unparsed scores reward 0 (training convention); Brier counts parsed only
        reward = mean([1.0 - (r["p"] - r["p_mc"]) ** 2 if r.get("p") is not None else 0.0
                       for r in rs])
        out[split] = {
            "n_calls": n, "parse_rate": len(parsed) / n,
            "reward_train_convention": reward,
            "brier_vs_pmc": mean([(r["p"] - r["p_mc"]) ** 2 for r in parsed]),
            "brier_vs_hard": mean([(r["p"] - r["hard"]) ** 2 for r in parsed
                                   if r["hard"] is not None]),
            "mean_p": mean([r["p"] for r in parsed]),
            "sd_p": (mean([(r["p"] - mean([x["p"] for x in parsed])) ** 2
                           for r in parsed]) or 0) ** 0.5,
            "truncated": sum(1 for r in rs if r.get("finish") == "length"),
            "errors": sum(1 for r in rs if r.get("error")),
        }
        by_tpl = defaultdict(list)
        for r in parsed:
            by_tpl[r["template_id"]].append((r["p"] - r["p_mc"]) ** 2)
        out[split]["brier_by_template"] = {k: sum(v) / len(v)
                                           for k, v in sorted(by_tpl.items())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples-train", type=int, default=1)
    ap.add_argument("--samples-val", type=int, default=4)
    ap.add_argument("--limit-train", type=int, default=0, help="0 = all")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = load_rows()
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    if args.limit_train:
        random.Random(0).shuffle(train)
        train = train[:args.limit_train]
    jobs = [(r, i) for r in train for i in range(args.samples_train)]
    jobs += [(r, i) for r in val for i in range(args.samples_val)]
    print(f"[{args.tag}] {len(train)} train rows x{args.samples_train}, "
          f"{len(val)} val rows x{args.samples_val} -> {len(jobs)} calls")

    results = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(eval_one, args.model, r, i) for r, i in jobs]
        for k, f in enumerate(cf.as_completed(futs)):
            results.append(f.result())
            if (k + 1) % 50 == 0:
                el = time.time() - t0
                print(f"  {k+1}/{len(jobs)} ({el:.0f}s, "
                      f"{sum(1 for r in results if r.get('p') is None)} unparsed)",
                      flush=True)

    summary = summarize(results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model, "tag": args.tag, "summary": summary,
               "results": results}, open(args.output, "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk != "brier_by_template"}
                      for k, v in summary.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
