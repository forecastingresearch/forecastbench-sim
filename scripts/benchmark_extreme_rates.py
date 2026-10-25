#!/usr/bin/env python3
"""Benchmark models on questions with extreme base rates."""

import json
import random
import re
import os
from pathlib import Path

from utils.llm.model_registry import configure_api_keys, MODELS
from dotenv import load_dotenv

def load_questions_by_template(data_dir: Path, template_filters: dict[str, str | None]) -> dict[str, list]:
    """Load questions filtered by template_id and optionally by horizon.

    Args:
        template_filters: dict mapping template_id -> horizon (or None for all horizons)
    """
    questions = {t: [] for t in template_filters}
    for game_dir in sorted(data_dir.iterdir()):
        if not game_dir.is_dir() or game_dir.name.startswith('.'):
            continue
        qfile = game_dir / "questions.json"
        if not qfile.exists():
            continue
        with open(qfile) as f:
            data = json.load(f)
        game_id = data.get("game_id", game_dir.name)
        for q in data.get("questions", []):
            tid = q.get("template_id")
            if tid in template_filters:
                horizon = q.get("difficulty", {}).get("horizon", "?")
                required_horizon = template_filters[tid]
                # Filter by horizon if specified
                if required_horizon is not None and horizon != required_horizon:
                    continue
                res = q.get("resolution", {})
                answer = res.get("answer")
                if answer is not None:
                    questions[tid].append({
                        "game_id": game_id,
                        "question_id": q.get("question_id"),
                        "question_text": q.get("question_text"),
                        "ground_truth": bool(answer),
                        "horizon": horizon,
                    })
    return questions

def load_world_report(data_dir: Path, game_id: str) -> str:
    report_path = data_dir / game_id / "world_report" / "turn_050_data.json"
    if not report_path.exists():
        return ""
    with open(report_path) as f:
        return json.dumps(json.load(f), separators=(',', ':'))

def build_prompt(question_text: str, world_report: str) -> str:
    return f"""You are a forecaster analyzing a FreeCiv game. Based on the world report data below, estimate the probability (0.0 to 1.0) that the following will resolve to YES.

## World Report Data (Turn 50)
{world_report}

## Question
{question_text}

Respond with ONLY a number between 0.0 and 1.0."""

def parse_prob(resp: str) -> float | None:
    m = re.search(r'(\d+\.?\d*)', resp.strip())
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    return None

def query_model(model, prompt: str) -> float | None:
    try:
        resp = model.get_response(prompt, temperature=0.0, max_tokens=50)
        return parse_prob(resp)
    except Exception as e:
        print(f"    Error: {e}")
        return None

def brier(preds, outcomes):
    if not preds:
        return float('nan')
    return sum((p - int(o))**2 for p, o in zip(preds, outcomes)) / len(preds)

def main():
    # Load env from project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    configure_api_keys(from_gcp=True)
    print("API keys configured")

    data_dir = Path(__file__).parent.parent / "data" / "questions"
    # Filter: wonder_first H1 (lower base rate), city_conquered_any H2 (higher base rate)
    template_filters = {
        "wonder_first": "H1",
        "city_conquered_any": "H2",
    }

    print("Loading questions...")
    all_qs = load_questions_by_template(data_dir, template_filters)
    for t, qs in all_qs.items():
        true_ct = sum(1 for q in qs if q["ground_truth"])
        horizon = template_filters[t]
        print(f"  {t} ({horizon}): {len(qs)} questions, base rate = {true_ct/len(qs):.1%}")

    # Sample 25 from each
    random.seed(42)
    samples = {}
    for t, qs in all_qs.items():
        samples[t] = random.sample(qs, min(25, len(qs)))
        print(f"  Sampled {len(samples[t])} {t} questions")

    # Models to test
    model_ids = ['gpt-4o-mini', 'claude-sonnet-4-5-20250929']
    models = {}
    for mid in model_ids:
        for m in MODELS:
            if mid in m.id or m.id in mid:
                models[mid] = m
                break

    print(f"\nModels: {list(models.keys())}")

    # Evaluate
    results = {}
    for template, qs in samples.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {template}")
        print(f"{'='*60}")

        results[template] = {"questions": [], "model_scores": {}}

        for i, q in enumerate(qs):
            print(f"\n[{i+1}/{len(qs)}] {q['game_id']} - {q['question_text'][:60]}...")
            print(f"  Truth: {q['ground_truth']} | Horizon: {q['horizon']}")

            world_report = load_world_report(data_dir, q["game_id"])
            if not world_report:
                print("  [No world report, skipping]")
                continue

            prompt = build_prompt(q["question_text"], world_report)
            preds = {}
            for mid, model in models.items():
                p = query_model(model, prompt)
                if p is not None:
                    preds[mid] = p
                    print(f"  {mid}: {p:.2f}")

            results[template]["questions"].append({
                "game_id": q["game_id"],
                "ground_truth": q["ground_truth"],
                "predictions": preds,
            })

        # Compute Brier scores
        print(f"\n--- {template} Results ---")
        base_rate = sum(1 for q in qs if q["ground_truth"]) / len(qs)

        for mid in models:
            preds = [qr["predictions"].get(mid) for qr in results[template]["questions"] if mid in qr["predictions"]]
            outcomes = [qr["ground_truth"] for qr in results[template]["questions"] if mid in qr["predictions"]]
            bs = brier(preds, outcomes)
            results[template]["model_scores"][mid] = bs
            print(f"  {mid}: Brier = {bs:.4f} (n={len(preds)})")

        # Baseline
        outcomes_all = [qr["ground_truth"] for qr in results[template]["questions"]]
        baseline_bs = brier([base_rate]*len(outcomes_all), outcomes_all)
        print(f"  Baseline (always {base_rate:.2f}): Brier = {baseline_bs:.4f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for template, horizon in template_filters.items():
        print(f"\n{template} ({horizon}):")
        for mid, score in results[template]["model_scores"].items():
            print(f"  {mid}: {score:.4f}")

if __name__ == "__main__":
    main()
