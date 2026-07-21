#!/usr/bin/env python3
"""Multi-turn natural-conditional elicitation.

Per question (per sample): one baseline exchange —
  user: OF binary prompt (world report + question)  ->  assistant: p̂(Y)
then, branching from that SAME exchange, one second turn per conditioning event:
  user: "Now suppose that <event> occurs during turns a-b. ... forecast again"
  ->  assistant: p̂(Y|X)

The model's own baseline answer stays in context (true multi-turn updating).

Usage:
  set -a; source .env; set +a
  uv run python scripts/uplift_v2/natcond_elicit.py \
      --cells tmp/natcond/seed2_cells.json --game-id seed2 \
      --model accounts/fireworks/models/gpt-oss-120b --tag oss120 \
      --samples 2 --concurrency 12 --output tmp/natcond/elicit_seed2_oss120.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_of_dataset import OF_BINARY_PROMPT  # noqa: E402
from eval_a1_gate import parse_prob  # noqa: E402
import eval_a1_gate  # noqa: E402

COND_TEMPLATE = (
    "Now consider a conditional scenario: suppose that the following occurs "
    "during turns {a}-{b} of this game: {event}. "
    "Given this additional assumption (and everything you already know), "
    "provide an updated forecast for the SAME question. Reason briefly about "
    "how, if at all, this event changes the outlook, then end with your "
    "updated probability in <probability> </probability> tags.")


def chat(model, messages):
    """Provider-agnostic chat. Env:
    CHAT_BASE_URL (default fireworks), CHAT_API_KEY (default FIREWORKS_API_KEY),
    CHAT_PROVIDER=openai|anthropic (default openai-compatible),
    CHAT_MAX_TOKENS (default: omitted for openai-compat, 16000 for anthropic)."""
    import os, urllib.request
    provider = os.environ.get("CHAT_PROVIDER", "openai")
    key = os.environ.get("CHAT_API_KEY") or os.environ["FIREWORKS_API_KEY"]
    mt = os.environ.get("CHAT_MAX_TOKENS")
    if provider == "anthropic":
        body = {"model": model, "messages": messages,
                "max_tokens": int(mt or 16000), "temperature": 1.0}
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode(),
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json",
                     "User-Agent": "civbench-natcond/1.0"})
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        return "".join(b.get("text", "") for b in d.get("content", []))
    base = os.environ.get("CHAT_BASE_URL",
                          "https://api.fireworks.ai/inference/v1")
    body = {"model": model, "messages": messages, "temperature": 1.0}
    if mt:
        body["max_tokens"] = int(mt)
    elif "fireworks" in base:
        body["max_tokens"] = 8192  # fireworks default is tiny; keep old behavior
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "User-Agent": "civbench-natcond/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"] or ""


def build_prompt(report, question, rt):
    background = ("This is a partial report on a FreeCiv game simulation in "
                  "progress, observed at turn 60. Five AI civilizations are "
                  "competing.\n\n" + report)
    rc = (f"Resolves YES if the answer to the question is affirmative in the "
          f"simulation state at turn {rt}, as determined by the game's "
          f"recorded metrics.")
    return OF_BINARY_PROMPT.format(question_title=question,
                                   background=background,
                                   resolution_criteria=rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cells = json.load(open(args.cells))
    report = Path(f"data/questions_mc/{args.game_id}/world_report/"
                  f"turn_060_report.txt").read_text()
    report = re.sub(r"^={10,} WORLD REPORT.*?\n|={10,} END REPORT.*$", "",
                    report, flags=re.DOTALL).strip()

    by_q = defaultdict(list)
    for c in cells:
        by_q[c["qid"]].append(c)
    qtext = {c["qid"]: c["question"] for c in cells}
    print(f"[{args.tag}] {len(by_q)} questions x {args.samples} samples "
          f"baselines, {len(cells)} cells x {args.samples} conditionals")

    results = []
    lock_print = time.time()

    def run_question(qid, sample):
        out = []
        base_prompt = build_prompt(report, qtext[qid], 90)
        for attempt in range(3):
            try:
                base_ans = chat(args.model, [
                    {"role": "user", "content": base_prompt}])
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return [{"qid": qid, "sample": sample, "stage": "base",
                             "p": None, "error": str(e)[:150]}]
                time.sleep(5 * (attempt + 1))
        p_base = parse_prob(base_ans)
        out.append({"qid": qid, "sample": sample, "stage": "base",
                    "event_id": None, "p": p_base})
        base_msgs = [{"role": "user", "content": base_prompt},
                     {"role": "assistant", "content": base_ans}]
        for c in by_q[qid]:
            a, b = c["window"]
            cond = COND_TEMPLATE.format(a=a, b=b, event=c["event_desc"])
            p_cond, err = None, None
            for attempt in range(3):
                try:
                    ans = chat(args.model, base_msgs +
                               [{"role": "user", "content": cond}])
                    p_cond = parse_prob(ans)
                    break
                except Exception as e:  # noqa: BLE001
                    err = str(e)[:150]
                    time.sleep(5 * (attempt + 1))
            out.append({"qid": qid, "sample": sample, "stage": "cond",
                        "event_id": c["event_id"], "p": p_cond,
                        **({"error": err} if err and p_cond is None else {})})
        return out

    jobs = [(q, s) for q in by_q for s in range(args.samples)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_question, q, s) for q, s in jobs]
        for k, f in enumerate(cf.as_completed(futs)):
            results.extend(f.result())
            print(f"  question-block {k+1}/{len(jobs)} done "
                  f"({time.time()-t0:.0f}s, {len(results)} rows)", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model, "tag": args.tag, "cells": args.cells,
               "results": results}, open(args.output, "w"), indent=1)
    bad = sum(1 for r in results if r.get("p") is None)
    print(f"done: {len(results)} rows, {bad} unparsed -> {args.output}")


if __name__ == "__main__":
    main()
