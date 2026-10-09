#!/usr/bin/env python3
"""
Mine a low-probability question corpus from the recorded FreeCiv games.

For the Red Lines / TailRiskBench MVP: a single recorded game gives a
deterministic 0/1 resolution, so the *true low probability* of a question class
is its cross-game base rate. We group binary forecasting questions by
(template_id, target-item, horizon), compute the base rate across all games,
and select classes whose base rate falls in a target tail band (default 1-9%)
with adequate support. Each game contributes one Bernoulli draw against the
class base rate -> real ground-truth calibration data in the tail.

Usage:
    uv run python worlds/freeciv/scripts/mine_lowprob_corpus.py \
        --data-dir data/games --snapshot-turn 40 \
        --rate-lo 0.01 --rate-hi 0.09 --min-n 40 --workers 8 \
        --out-dir data/lowprob
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, 'src')


def _target_key(template_id: str, params: dict) -> str:
    """Item that makes a question class specific (and its rareness meaningful)."""
    for k in ("tech_name", "wonder_name", "government_type"):
        if k in params and params[k] is not None:
            return f"{k}={params[k]}"
    return ""  # event/rank templates: aggregate by (template, horizon)


def process_single_game(args: tuple):
    """Generate + resolve binary forecasting questions for one game.

    Returns a compact list of records (no bulky world data), or None.
    """
    data_file, snapshot_turn = args
    try:
        sys.path.insert(0, 'src')
        from freeciv_world.world_reports.questions import (
            QuestionGenerator, QuestionResolver, REGISTRY, question_bank_to_dict,
        )
        with open(data_file) as f:
            game_data = json.load(f)

        game_id = game_data.get('metadata', {}).get('username', Path(data_file).stem)
        max_turn = game_data.get('metadata', {}).get('turn', 0)
        if snapshot_turn >= max_turn:
            return None

        generator = QuestionGenerator()
        resolver = QuestionResolver(REGISTRY)

        # H1-H7 binary forecasting questions only (skip H0 comprehension + continuous)
        bank = generator.generate_question_bank(
            game_id=game_id, game_data=game_data, snapshot_turn=snapshot_turn,
        )
        resolved = resolver.resolve_batch(bank, game_data)
        qd = question_bank_to_dict(resolved)

        out = []
        for q in qd.get('questions', []):
            ans = (q.get('resolution') or {}).get('answer')
            if not isinstance(ans, bool):
                continue
            params = q.get('parameters', {})
            out.append({
                "game_id": game_id,
                "question_id": q.get("question_id"),
                "template_id": q.get("template_id"),
                "horizon": q.get("horizon"),
                "target": _target_key(q.get("template_id", ""), params),
                "resolution_turn": q.get("resolution_turn"),
                "snapshot_turn": snapshot_turn,
                "question_text": q.get("question_text"),
                "ground_truth": bool(ans),
            })
        return out
    except Exception as e:
        print(f"Error processing {data_file}: {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser(description="Mine low-probability FreeCiv question corpus")
    ap.add_argument("--data-dir", default="data/games")
    ap.add_argument("--snapshot-turn", type=int, default=40)
    ap.add_argument("--rate-lo", type=float, default=0.01)
    ap.add_argument("--rate-hi", type=float, default=0.09)
    ap.add_argument("--min-n", type=int, default=40,
                    help="min games supporting a class (40 = the reference run; "
                         "recorded in lowprob_classes.json meta.min_n)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="data/lowprob")
    ap.add_argument("--limit", type=int, default=None, help="cap #games (debug)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*_data.json"))
    if args.limit:
        files = files[: args.limit]
    print(f"Found {len(files)} game files; snapshot turn {args.snapshot_turn}")

    task_args = [(str(f), args.snapshot_turn) for f in files]
    records = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_single_game, a): a[0] for a in task_args}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if r:
                records.extend(r)
            if done % 100 == 0:
                print(f"  {done}/{len(files)} games, {len(records)} binary questions so far")

    print(f"Total binary questions: {len(records)}")

    # Group into classes: (template, target, horizon)
    classes = defaultdict(list)  # key -> list of records
    for rec in records:
        key = (rec["template_id"], rec["target"], rec["horizon"])
        classes[key].append(rec)

    class_rows = []
    for (tmpl, target, horizon), recs in classes.items():
        n = len(recs)
        yes = sum(1 for r in recs if r["ground_truth"])
        rate = yes / n if n else 0.0
        class_rows.append({
            "template_id": tmpl, "target": target, "horizon": horizon,
            "n": n, "yes": yes, "base_rate": rate,
        })
    class_rows.sort(key=lambda x: x["base_rate"])

    # Select tail classes
    selected = [
        c for c in class_rows
        if c["n"] >= args.min_n and args.rate_lo <= c["base_rate"] <= args.rate_hi
    ]
    selected_keys = {(c["template_id"], c["target"], c["horizon"]) for c in selected}

    # Instance-level corpus for selected classes, tagging each with its class base rate (true prob)
    rate_by_key = {(c["template_id"], c["target"], c["horizon"]): c["base_rate"] for c in selected}
    corpus = []
    for rec in records:
        key = (rec["template_id"], rec["target"], rec["horizon"])
        if key in selected_keys:
            rec2 = dict(rec)
            rec2["class_base_rate"] = rate_by_key[key]  # true low probability of the class
            corpus.append(rec2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "snapshot_turn": args.snapshot_turn, "n_games": len(files),
        "rate_band": [args.rate_lo, args.rate_hi], "min_n": args.min_n,
        "total_binary_questions": len(records),
        "n_classes_total": len(class_rows),
        "n_classes_selected": len(selected),
        "n_corpus_instances": len(corpus),
    }
    (out_dir / "lowprob_classes.json").write_text(
        json.dumps({"meta": meta, "classes": class_rows}, indent=2))
    (out_dir / "lowprob_selected_classes.json").write_text(
        json.dumps({"meta": meta, "classes": selected}, indent=2))
    (out_dir / "lowprob_questions.json").write_text(
        json.dumps({"meta": meta, "questions": corpus}, indent=2))

    print("\n=== SUMMARY ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print(f"\nSelected {len(selected)} tail classes "
          f"({args.rate_lo:.0%}-{args.rate_hi:.0%}), {len(corpus)} instances. "
          f"Wrote to {out_dir}/")
    print("\nSelected classes (rate | n | template | horizon | target):")
    for c in sorted(selected, key=lambda x: x["base_rate"]):
        print(f"  {c['base_rate']:.3f}  n={c['n']:4d}  "
              f"{c['template_id']:20s} {c['horizon']:3s}  {c['target']}")


if __name__ == "__main__":
    main()
