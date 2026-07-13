#!/usr/bin/env python3
"""Step-0 baseline eval of small models on FB-Sim binary questions (Fireworks).

Evaluates candidate small instruct models on the MC-labeled question records in
data/training/{train,val}.jsonl (each record: prompt with world report + question,
p_mc dense target, baseline_01 hard label). Single-question calls — no batching —
with a format ladder tuned for ~zero parse failures:

  mode "line"   : instruct the model to end with "PROBABILITY: 0.XX"; parse the
                  LAST match; one retry with a terse reminder on failure.
  mode "schema" : Fireworks structured output (response_format json_schema with
                  {"probability": number}); parse is guaranteed JSON.

Reports per model: parse rate, Brier vs p_mc (dense), Brier vs baseline_01 (hard),
ECE (10-bin, vs hard), mean/SD of predictions, per-template Brier.

Usage:
  set -a; source .env; set +a
  uv run python scripts/uplift_v2/eval_small_fbsim.py \
      --model accounts/fireworks/models/qwen3-4b-instruct-2507 \
      --records data/training/train.jsonl data/training/val.jsonl \
      --limit 20 --mode line --output tmp/uplift_v2/probe_q4b.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import time
import urllib.request
from pathlib import Path

API_BASE = "https://api.fireworks.ai/inference/v1"

FORMAT_SUFFIX_LINE = (
    "\n\nGive a brief analysis (a few sentences at most), then end your response "
    "with a single final line in exactly this format:\nPROBABILITY: 0.XX\n"
    "where 0.XX is your probability estimate between 0.00 and 1.00. "
    "You MUST output that final line under all circumstances."
)
FORMAT_SUFFIX_SCHEMA = (
    "\n\nRespond with a JSON object containing your probability estimate between "
    "0.0 and 1.0, e.g. {\"probability\": 0.42}."
)
RETRY_REMINDER = (
    "Your previous response did not include the required final line. "
    "Reply now with ONLY the single line:\nPROBABILITY: 0.XX"
)

PROB_LINE_RE = re.compile(r"PROBABILITY:\s*([01]?\.\d+|[01](?:\.\d*)?)", re.IGNORECASE)


def call_chat(model: str, messages: list[dict], max_tokens: int, mode: str,
              timeout: int = 120) -> str:
    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if mode == "schema":
        body["response_format"] = {
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {"probability": {"type": "number"}},
                "required": ["probability"],
            },
        }
    req = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}",
                 "Content-Type": "application/json",
                 "User-Agent": "civbench-eval/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"] or ""


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def parse_prob(text: str, mode: str) -> float | None:
    text = strip_think(text)
    if mode == "schema":
        try:
            v = json.loads(text).get("probability")
            return float(v) if v is not None and 0.0 <= float(v) <= 1.0 else None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    matches = PROB_LINE_RE.findall(text)
    if not matches:
        return None
    try:
        v = float(matches[-1])
    except ValueError:
        return None
    return v if 0.0 <= v <= 1.0 else None


def eval_record(rec: dict, model: str, mode: str, max_tokens: int,
                no_think: bool) -> dict:
    suffix = FORMAT_SUFFIX_SCHEMA if mode == "schema" else FORMAT_SUFFIX_LINE
    content = rec["prompt"] + suffix
    if no_think:
        content = "/no_think\n" + content
    messages = [{"role": "user", "content": content}]
    t0 = time.time()
    result = {"qid": rec["qid"], "game_id": rec["game_id"],
              "template_id": rec["template_id"], "p_mc": float(rec["p_mc"]),
              "hard": int(rec["baseline_01"]), "pred": None, "retried": False,
              "error": None}
    try:
        text = call_chat(model, messages, max_tokens, mode)
        pred = parse_prob(text, mode)
        if pred is None and mode == "line":
            result["retried"] = True
            messages += [{"role": "assistant", "content": text[-2000:]},
                         {"role": "user", "content": RETRY_REMINDER}]
            text2 = call_chat(model, messages, 64, mode)
            pred = parse_prob(text2, mode)
            if pred is None:
                result["raw_tail"] = strip_think(text)[-300:]
        result["pred"] = pred
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"[:300]
    result["sec"] = round(time.time() - t0, 1)
    return result


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["pred"] is not None]
    n = len(rows)
    summ: dict = {"n": n, "n_parsed": len(ok), "parse_rate": len(ok) / n if n else 0,
                  "n_retried": sum(1 for r in rows if r["retried"]),
                  "n_errors": sum(1 for r in rows if r["error"])}
    if not ok:
        return summ
    preds = [r["pred"] for r in ok]
    summ["brier_dense"] = sum((r["pred"] - r["p_mc"]) ** 2 for r in ok) / len(ok)
    summ["brier_hard"] = sum((r["pred"] - r["hard"]) ** 2 for r in ok) / len(ok)
    mean = sum(preds) / len(preds)
    summ["pred_mean"] = mean
    summ["pred_sd"] = (sum((p - mean) ** 2 for p in preds) / len(preds)) ** 0.5
    # 10-bin ECE vs hard label
    bins: dict[int, list] = {}
    for r in ok:
        bins.setdefault(min(int(r["pred"] * 10), 9), []).append(r)
    summ["ece_hard"] = sum(
        abs(sum(x["pred"] for x in b) / len(b) - sum(x["hard"] for x in b) / len(b))
        * len(b) for b in bins.values()) / len(ok)
    per_t: dict[str, list] = {}
    for r in ok:
        per_t.setdefault(r["template_id"], []).append(r)
    summ["per_template_brier_dense"] = {
        t: round(sum((x["pred"] - x["p_mc"]) ** 2 for x in b) / len(b), 4)
        for t, b in sorted(per_t.items())}
    return summ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--mode", choices=["line", "schema"], default="line")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--no-think", action="store_true",
                    help="prepend Qwen3 /no_think soft switch")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    recs = []
    for path in args.records:
        with open(path) as f:
            recs += [json.loads(line) for line in f]
    if args.limit:
        recs = recs[:: max(1, len(recs) // args.limit)][:args.limit]  # stratified stride
    print(f"{args.model}: {len(recs)} records, mode={args.mode}, "
          f"max_tokens={args.max_tokens}, no_think={args.no_think}")

    rows = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(eval_record, r, args.model, args.mode, args.max_tokens,
                          args.no_think) for r in recs]
        for i, fut in enumerate(cf.as_completed(futs)):
            rows.append(fut.result())
            if (i + 1) % 25 == 0 or i + 1 == len(recs):
                s = summarize(rows)
                print(f"  [{i+1}/{len(recs)}] parse={s['parse_rate']:.3f} "
                      f"brier_dense={s.get('brier_dense', float('nan')):.4f} "
                      f"({time.time()-t0:.0f}s)")

    summ = summarize(rows)
    out = {"model": args.model, "config": vars(args), "summary": summ, "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in summ.items()
                      if k != "per_template_brier_dense"}, indent=1))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
