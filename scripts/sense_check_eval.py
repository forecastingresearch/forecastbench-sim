#!/usr/bin/env python3
"""Quick sense-check: evaluate Sonnet on a mix of all 4 question types.

Types:
  1. Binary unconditional
  2. Continuous unconditional
  3. Binary conditional
  4. Continuous conditional

Usage:
    uv run python scripts/sense_check_eval.py --dry-run   # Just show prompts
    uv run python scripts/sense_check_eval.py              # Actually call Sonnet
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_sample_questions(data_dir: Path, n_per_type: int = 3) -> list[dict]:
    """Load a small sample of each question type."""
    questions = []

    # 1. Binary + continuous unconditional from questions_all.json
    combined = data_dir / "questions_all.json"
    if combined.exists():
        data = json.load(open(combined))
        binary_uncon = []
        continuous_uncon = []
        for q in data.get("questions", []):
            res = q.get("resolution", {})
            if res.get("answer") is None and res.get("value") is None:
                continue
            qt = q.get("question_type", "binary")
            game_id = q.get("parameters", {}).get("game_id", "seed0")
            entry = {
                "game_id": game_id,
                "question_id": q["question_id"],
                "template_id": q.get("template_id", ""),
                "question_text": q.get("question_text", ""),
                "question_type": qt,
                "ground_truth": res.get("value") if qt == "continuous" else bool(res.get("answer")),
                "category": f"{qt}_unconditional",
            }
            if qt == "continuous" and len(continuous_uncon) < n_per_type:
                continuous_uncon.append(entry)
            elif qt == "binary" and len(binary_uncon) < n_per_type:
                binary_uncon.append(entry)
            if len(binary_uncon) >= n_per_type and len(continuous_uncon) >= n_per_type:
                break
        questions.extend(binary_uncon)
        questions.extend(continuous_uncon)

    # 2. Binary + continuous conditional from conditional_results.json
    cond_results_path = Path("logs/recordings/seed0forkgoldadd500p0/conditional_results.json")
    if cond_results_path.exists():
        cond_data = json.load(open(cond_results_path))
        binary_cond = []
        continuous_cond = []
        for cq in cond_data.get("questions", []):
            cid = cq["conditional_id"]
            result = cond_data.get("results", {}).get(cid, {})
            is_continuous = cq["target_template_id"].endswith("_continuous")

            # Build question text using condition + target
            cond = cq.get("condition", {})
            civ = cq["target_parameters"].get("civ", "the civilization")
            cond_desc = cond.get("description", "")
            res_turn = cq["resolution_turn"]
            template = cq["target_template_id"]

            if is_continuous:
                metric_names = {
                    "techs_continuous": "technologies",
                    "treasury_continuous": "gold",
                    "population_continuous": "population",
                    "cities_count_continuous": "cities",
                    "territory_continuous": "tiles",
                    "scores_continuous": "score points",
                }
                metric = metric_names.get(template, "units")
                q_text = f"If {cond_desc}, how many {metric} would {civ} have at turn {res_turn}?"
                gt = result.get("answer_intervention")
            else:
                civ_b = cq["target_parameters"].get("civ_b", "the other civilization")
                q_text = f"If {cond_desc}, would {civ} have more than {civ_b} at turn {res_turn}?"
                gt = result.get("answer_intervention")

            if gt is None:
                continue

            entry = {
                "game_id": "seed0",
                "question_id": f"{cid}_intervention",
                "template_id": f"conditional_{template}",
                "question_text": q_text,
                "question_type": "continuous" if is_continuous else "binary",
                "ground_truth": gt,
                "category": f"{'continuous' if is_continuous else 'binary'}_conditional",
            }

            if is_continuous and len(continuous_cond) < n_per_type:
                continuous_cond.append(entry)
            elif not is_continuous and len(binary_cond) < n_per_type:
                binary_cond.append(entry)
            if len(binary_cond) >= n_per_type and len(continuous_cond) >= n_per_type:
                break
        questions.extend(binary_cond)
        questions.extend(continuous_cond)

    return questions


def build_prompt(question: dict, world_report: str) -> str:
    """Build an appropriate prompt based on question type."""
    qt = question["question_type"]
    q_text = question["question_text"]

    if qt == "continuous":
        return f"""You are an expert superforecaster analyzing a FreeCiv game simulation.

## World Report
{world_report}

## Question
{q_text}

Provide your estimate as a set of percentiles representing your uncertainty distribution.
You MUST respond with exactly 5 numbers in this format:

<<<PERCENTILES>>>
p10: <your 10th percentile estimate>
p25: <your 25th percentile estimate>
p50: <your median estimate>
p75: <your 75th percentile estimate>
p90: <your 90th percentile estimate>
<<<END>>>

These should represent your belief about the range of likely values.
p10 = value you think there's only a 10% chance the true value is below.
p90 = value you think there's only a 10% chance the true value is above."""
    else:
        return f"""You are an expert superforecaster analyzing a FreeCiv game simulation.

## World Report
{world_report}

## Question
{q_text}

You MUST give a probability estimate between 0 and 1.
End your response with:

<<<PROBABILITY>>>
0.65
<<<END>>>

Replace 0.65 with your actual probability estimate."""


def parse_binary_response(response: str) -> float | None:
    """Parse probability from binary question response."""
    match = re.search(r'<<<PROBABILIT(?:Y|IES)>>>(.*?)<<<END>>>', response, re.DOTALL | re.IGNORECASE)
    if match:
        num = re.search(r'(\d*\.?\d+)', match.group(1))
        if num:
            return max(0.0, min(1.0, float(num.group(1))))
    # Fallback
    match = re.search(r'(\d*\.\d+)', response)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return None


def parse_continuous_response(response: str) -> dict | None:
    """Parse percentile estimates from continuous question response."""
    match = re.search(r'<<<PERCENTILES?>>>(.*?)<<<END>>>', response, re.DOTALL | re.IGNORECASE)
    text = match.group(1) if match else response

    percentiles = {}
    for label in ["p10", "p25", "p50", "p75", "p90"]:
        pattern = rf'{label}\s*[:=]\s*(-?\d+\.?\d*)'
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            percentiles[label] = float(m.group(1))

    return percentiles if len(percentiles) == 5 else None


async def query_sonnet(prompt: str) -> str:
    """Query Sonnet via litellm."""
    from litellm import acompletion

    response = await acompletion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2000,
    )
    return response.choices[0].message.content


async def main():
    parser = argparse.ArgumentParser(description="Sense-check eval across all question types")
    parser.add_argument("--dry-run", action="store_true", help="Just show prompts, don't call API")
    parser.add_argument("--n", type=int, default=2, help="Questions per type (default: 2)")
    parser.add_argument("--data-dir", type=str, default="data/questions", help="Questions directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("Loading sample questions...")
    questions = load_sample_questions(data_dir, n_per_type=args.n)

    # Group by category
    by_cat = {}
    for q in questions:
        cat = q["category"]
        by_cat.setdefault(cat, []).append(q)

    print(f"\nLoaded {len(questions)} questions:")
    for cat, qs in sorted(by_cat.items()):
        print(f"  {cat}: {len(qs)}")

    # Load world report for seed0
    world_report = ""
    wr_path = data_dir / "seed0" / "world_report" / "turn_060_report.txt"
    if wr_path.exists():
        world_report = open(wr_path).read()
        print(f"\nWorld report loaded: {len(world_report)} chars")
    else:
        print(f"\nWarning: No world report at {wr_path}")

    if not args.dry_run:
        # Load API keys from GCP
        try:
            from civrealm.evaluation.models import load_api_keys_from_gcp
            load_api_keys_from_gcp()
            print("API keys loaded from GCP")
        except Exception as e:
            print(f"Warning: Could not load GCP keys: {e}")
            if not os.environ.get("ANTHROPIC_API_KEY"):
                print("No ANTHROPIC_API_KEY available. Use --dry-run or set the key.")
                return 1

    print(f"\n{'='*70}")
    print("SENSE CHECK EVALUATION")
    print(f"{'='*70}")

    results = []
    for i, q in enumerate(questions):
        print(f"\n--- [{i+1}/{len(questions)}] {q['category']} ---")
        print(f"  ID: {q['question_id']}")
        print(f"  Template: {q['template_id']}")
        print(f"  Q: {q['question_text'][:100]}...")
        print(f"  Ground truth: {q['ground_truth']}")

        prompt = build_prompt(q, world_report)

        if args.dry_run:
            # Show prompt summary
            print(f"  Prompt length: {len(prompt)} chars")
            # Show the tail of prompt
            lines = prompt.strip().split('\n')
            print(f"  Prompt format (last 5 lines):")
            for line in lines[-5:]:
                print(f"    {line}")
            results.append({"question": q, "status": "dry_run"})
            continue

        # Actually call Sonnet
        try:
            response = await query_sonnet(prompt)
            print(f"  Response ({len(response)} chars): {response[:150]}...")

            if q["question_type"] == "continuous":
                parsed = parse_continuous_response(response)
                if parsed:
                    print(f"  Parsed percentiles: {parsed}")
                    print(f"  Median vs truth: {parsed.get('p50', '?')} vs {q['ground_truth']}")
                    results.append({"question": q, "parsed": parsed, "status": "ok"})
                else:
                    print(f"  PARSE FAILED")
                    results.append({"question": q, "raw": response[:300], "status": "parse_failed"})
            else:
                parsed = parse_binary_response(response)
                if parsed is not None:
                    print(f"  Parsed probability: {parsed:.3f}")
                    gt = q["ground_truth"]
                    brier = (parsed - (1.0 if gt else 0.0)) ** 2
                    print(f"  Brier: {brier:.3f}")
                    results.append({"question": q, "probability": parsed, "brier": brier, "status": "ok"})
                else:
                    print(f"  PARSE FAILED")
                    results.append({"question": q, "raw": response[:300], "status": "parse_failed"})

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"question": q, "error": str(e), "status": "error"})

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for cat in sorted(by_cat.keys()):
        cat_results = [r for r in results if r["question"]["category"] == cat]
        ok = sum(1 for r in cat_results if r["status"] == "ok")
        failed = sum(1 for r in cat_results if r["status"] == "parse_failed")
        errors = sum(1 for r in cat_results if r["status"] == "error")
        print(f"\n{cat}: {ok} ok, {failed} parse failures, {errors} errors")

        for r in cat_results:
            if r["status"] == "ok":
                if "probability" in r:
                    print(f"  {r['question']['question_id']}: prob={r['probability']:.3f}, brier={r['brier']:.3f}")
                elif "parsed" in r:
                    p = r["parsed"]
                    gt = r["question"]["ground_truth"]
                    print(f"  {r['question']['question_id']}: p50={p.get('p50','?')}, truth={gt}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
