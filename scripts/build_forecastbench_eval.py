#!/usr/bin/env python3
"""Build a clean real-world forecasting eval set from ForecastBench dumps.

Pulls every resolution_set + question_set from forecastingresearch/forecastbench-datasets,
joins them, filters to (a) market-style natural-language binary questions, (b) resolved
to 0 or 1, (c) freeze_datetime AFTER --cutoff (so the model couldn't have memorized the
answer). Writes a single CSV ready for prompting.

Usage:
    uv run python scripts/build_forecastbench_eval.py \\
        --cutoff 2025-01-01 \\
        --output data/forecastbench/eval.csv
"""
from __future__ import annotations

import sys
import json
import csv
import argparse
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_API = "https://api.github.com/repos/forecastingresearch/forecastbench-datasets/contents"
RAW = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main"

# Market-style sources (natural-language binary). Skip "dataset" sources
# (acled / fred / yfinance / wikipedia / dbnomics) which are binarised
# time-series and less natural for LLM forecasting transfer.
MARKET_SOURCES = {"manifold", "metaculus", "polymarket", "infer"}


def get_json(url: str) -> dict | list:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def parse_dt(s: str | None) -> datetime | None:
    if not s or s == "N/A":
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except Exception:
            return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True,
                    help="Model knowledge cutoff (YYYY-MM-DD). Keep questions whose "
                         "freeze_datetime is strictly after this date.")
    ap.add_argument("--output", default="data/forecastbench/eval.csv")
    ap.add_argument("--cache-dir", default="data/forecastbench/cache")
    ap.add_argument("--sources", nargs="+", default=sorted(MARKET_SOURCES),
                    help="Which sources to include (default: market-style).")
    args = ap.parse_args()

    cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d")
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    print(f"Listing resolution sets in forecastbench-datasets...")
    res_files = [
        x["name"] for x in get_json(f"{REPO_API}/datasets/resolution_sets")
        if x["name"].endswith("_resolution_set.json")
    ]
    print(f"  {len(res_files)} resolution sets")

    rows = []
    skipped = {"unresolved": 0, "non_binary": 0, "wrong_source": 0,
               "no_freeze": 0, "freeze_before_cutoff": 0, "missing_question": 0}

    for fname in sorted(res_files):
        date_str = fname.split("_")[0]
        res_path = cache / fname
        if not res_path.exists():
            res_path.write_text(json.dumps(get_json(f"{RAW}/datasets/resolution_sets/{fname}")))
        res = json.loads(res_path.read_text())
        qset_name = res["question_set"]
        q_path = cache / qset_name
        if not q_path.exists():
            q_path.write_text(json.dumps(get_json(f"{RAW}/datasets/question_sets/{qset_name}")))
        qset = json.loads(q_path.read_text())
        # Skip "combo" questions where id is a list of two ids (joint 2x2 forecasts).
        qmap = {(q["id"], q["source"]): q for q in qset["questions"]
                if isinstance(q.get("id"), str)}

        for r in res["resolutions"]:
            if not r.get("resolved"):
                skipped["unresolved"] += 1
                continue
            if r.get("source") not in args.sources:
                skipped["wrong_source"] += 1
                continue
            rt = r.get("resolved_to")
            if rt not in (0, 1, 0.0, 1.0):
                skipped["non_binary"] += 1
                continue
            if not isinstance(r.get("id"), str):
                skipped["missing_question"] += 1
                continue
            q = qmap.get((r["id"], r["source"]))
            if q is None:
                skipped["missing_question"] += 1
                continue
            freeze = parse_dt(q.get("freeze_datetime"))
            if freeze is None:
                skipped["no_freeze"] += 1
                continue
            if freeze <= cutoff:
                skipped["freeze_before_cutoff"] += 1
                continue
            rows.append({
                "qid": r["id"],
                "source": r["source"],
                "freeze_date": freeze.strftime("%Y-%m-%d"),
                "resolution_date": r.get("resolution_date") or "",
                "resolved_to": float(rt),
                "freeze_market_value": q.get("freeze_datetime_value") or "",
                "question": q.get("question") or "",
                "background": (q.get("background") or "").replace("\n", " ").strip(),
                "resolution_criteria": (q.get("resolution_criteria") or "").replace("\n", " ").strip(),
                "url": q.get("url") or "",
                "forecast_due_date": res["forecast_due_date"],
            })

    # dedupe by (qid, source) — same question can appear in multiple forecast rounds
    by_key = {}
    for row in rows:
        k = (row["qid"], row["source"])
        prev = by_key.get(k)
        # Prefer the earliest freeze (gives the model less info)
        if prev is None or row["freeze_date"] < prev["freeze_date"]:
            by_key[k] = row
    unique = list(by_key.values())

    print(f"\nKept {len(unique)} unique post-cutoff binary market questions "
          f"(from {sum(1 for _ in rows)} total resolutions across {len(res_files)} sets).")
    print("Skipped:")
    for k, v in skipped.items():
        print(f"  {k}: {v}")

    from collections import Counter
    by_src = Counter(r["source"] for r in unique)
    print(f"By source: {dict(by_src)}")
    by_year = Counter(r["resolution_date"][:4] for r in unique)
    print(f"By resolution year: {dict(by_year)}")
    base_rate = sum(r["resolved_to"] for r in unique) / max(1, len(unique))
    print(f"Base rate (P(yes)): {base_rate:.3f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(unique[0].keys()))
        w.writeheader()
        w.writerows(unique)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
