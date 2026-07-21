#!/usr/bin/env python3
"""A2 three-way eval: run a model over the RFT-format eval sets
(data/uplift_v2/rft_v2/{val,test}_dense.jsonl — dense file carries both targets
in metadata? No: dense ground_truth = p_mc; hard label comes from the matching
_hard file). Decoding matches training: temp 1.0, top_p 1.0, max_tokens 8192.

Usage:
  set -a; source .env; set +a
  uv run python scripts/uplift_v2/eval_a2.py \
      --model 'accounts/jaeholee/models/<m>#accounts/jaeholee/deployments/<d>' \
      --tag dense-s1 --sets val test --samples-val 2 --samples-test 1 \
      --concurrency 16 --output tmp/uplift_v2/a2_dense-s1.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_a1_gate import call_chat, parse_prob, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RFT = ROOT / "data/uplift_v2/rft_v2"


def load_set(name: str) -> list[dict]:
    dense = [json.loads(l) for l in open(RFT / f"{name}_dense.jsonl")]
    hard = [json.loads(l) for l in open(RFT / f"{name}_hard.jsonl")]
    hkey = {(r["metadata"]["game_id"], r["metadata"]["qid"]):
            float(r["ground_truth"]) for r in hard}
    rows = []
    for r in dense:
        md = r["metadata"]
        rows.append({
            "prompt": r["messages"][0]["content"],
            "p_mc": float(r["ground_truth"]),
            "hard": hkey[(md["game_id"], md["qid"])],
            "qid": md["qid"], "game_id": md["game_id"],
            "template_id": md["template_id"], "split": name,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", nargs="+", default=["val", "test"])
    ap.add_argument("--samples-val", type=int, default=2)
    ap.add_argument("--samples-test", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    jobs = []
    for s in args.sets:
        n = args.samples_val if s == "val" else args.samples_test
        for row in load_set(s):
            for i in range(n):
                jobs.append((row, i))
    print(f"[{args.tag}] {len(jobs)} calls over {args.sets}")

    def work(job):
        row, i = job
        for attempt in range(3):
            try:
                out = call_chat(args.model, row["prompt"])
                return {**{k: row[k] for k in
                           ("qid", "game_id", "template_id", "split",
                            "p_mc", "hard")},
                        "sample": i, "p": parse_prob(out["text"]),
                        "finish": out["finish"],
                        "completion_tokens":
                            out["usage"].get("completion_tokens")}
            except Exception as e:  # noqa: BLE001
                err = str(e)[:200]
                time.sleep(5 * (attempt + 1))
        return {**{k: row[k] for k in
                   ("qid", "game_id", "template_id", "split", "p_mc", "hard")},
                "sample": i, "p": None, "error": err}

    results = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for k, f in enumerate(cf.as_completed(futs)):
            results.append(f.result())
            if (k + 1) % 100 == 0:
                bad = sum(1 for r in results if r.get("p") is None)
                print(f"  {k+1}/{len(jobs)} ({time.time()-t0:.0f}s, "
                      f"{bad} unparsed)", flush=True)

    # summarize() groups by 'split' — patch its split list dynamically
    summary = {}
    for s in args.sets:
        sub = [r for r in results if r["split"] == s]
        import eval_a1_gate
        tmp = eval_a1_gate.summarize(
            [dict(r, split="train") for r in sub])  # reuse: label irrelevant
        summary[s] = tmp.get("train")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model, "tag": args.tag, "summary": summary,
               "results": results}, open(args.output, "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in (v or {}).items()
                          if kk != "brier_by_template"}
                      for k, v in summary.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
