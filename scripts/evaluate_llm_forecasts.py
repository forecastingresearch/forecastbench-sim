#!/usr/bin/env python3
"""
Evaluate LLMs on tech_comparative forecasting questions.

This script:
1. Loads questions from data/questions/tech_comparative/
2. Subsamples N questions with a configurable seed
3. Queries multiple LLMs for probability forecasts
4. Computes and reports Brier scores

Usage:
    python scripts/evaluate_llm_forecasts.py --seed 42 --num-questions 50

Requirements:
    pip install fri-utils python-dotenv
"""

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

from utils.llm.model_registry import configure_api_keys, MODELS

from dotenv import load_dotenv


def load_questions(data_dir: Path, template_filter: str = "tech_comparative") -> list[dict]:
    """
    Load all questions from game folders, filtering by template_id.

    Returns list of dicts with keys: game_id, question_id, question_text, ground_truth, parameters
    """
    questions = []

    for game_dir in sorted(data_dir.iterdir()):
        if not game_dir.is_dir():
            continue

        questions_file = game_dir / "questions.json"
        if not questions_file.exists():
            continue

        with open(questions_file) as f:
            data = json.load(f)

        game_id = data.get("game_id", game_dir.name)

        for q in data.get("questions", []):
            if template_filter and q.get("template_id") != template_filter:
                continue

            resolution = q.get("resolution", {})
            answer = resolution.get("answer")
            if answer is None:
                continue

            questions.append({
                "game_id": game_id,
                "question_id": q.get("question_id"),
                "question_text": q.get("question_text"),
                "ground_truth": bool(answer),
                "parameters": q.get("parameters", {}),
                "difficulty": q.get("difficulty", {}),
            })

    return questions


def load_world_report(data_dir: Path, game_id: str) -> str:
    """Load world report JSON data as raw JSON string."""
    report_path = data_dir / game_id / "world_report" / "turn_050_data.json"

    if not report_path.exists():
        return ""

    with open(report_path) as f:
        data = json.load(f)

    # Return as compact JSON string
    return json.dumps(data, separators=(',', ':'))


def subsample_questions(questions: list[dict], n: int, seed: int) -> list[dict]:
    """Subsample n questions with reproducible randomness."""
    random.seed(seed)
    if n >= len(questions):
        return questions
    return random.sample(questions, n)


def build_prompt(question_text: str, world_report: str) -> str:
    """Build the prompt for the LLM."""
    return f"""You are a forecaster analyzing a civilization simulation game. Based on the world report data (JSON) below, provide your probability estimate (a probability between 0.0 and 1.0) that the following question will resolve to YES.

The JSON contains time series data for each civilization, keyed by turn number and player ID.

## World Report Data (Turn 50)
{world_report}

## Question
{question_text}

Respond with ONLY a number between 0.0 and 1.0 representing your probability estimate."""


def parse_probability(response: str) -> float | None:
    """Parse a probability value from model response."""
    # Try to find a decimal number in the response
    match = re.search(r'(\d+\.?\d*)', response.strip())
    if match:
        value = float(match.group(1))
        # Clamp to [0, 1]
        return max(0.0, min(1.0, value))
    return None


def query_model(model, prompt: str, max_retries: int = 3) -> float | None:
    """Query a model and parse the probability response."""
    for attempt in range(max_retries):
        try:
            response = model.get_response(prompt, temperature=0.0, max_tokens=50)
            prob = parse_probability(response)
            if prob is not None:
                return prob
            print(f"  Warning: Could not parse response: {response[:100]}")
        except Exception as e:
            print(f"  Error querying model (attempt {attempt + 1}): {e}")
    return None


def compute_brier_score(predictions: list[float], outcomes: list[bool]) -> float:
    """
    Compute Brier score: mean squared error between predictions and outcomes.
    Lower is better. Perfect = 0.0, worst = 1.0.
    """
    if not predictions:
        return float('nan')
    return sum((p - int(o))**2 for p, o in zip(predictions, outcomes)) / len(predictions)


def get_model_by_id(models_list, model_id: str):
    """Find a model in the MODELS list by ID (partial match)."""
    for model in models_list:
        if model_id in model.id or model.id in model_id:
            return model
    return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate LLMs on forecasting questions')
    parser.add_argument('--data-dir', type=str, default='data/questions/tech_comparative',
                        help='Directory containing question data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num-questions', '-n', type=int, default=50,
                        help='Number of questions to sample')
    parser.add_argument('--output', '-o', type=str,
                        help='Output JSON file path')
    parser.add_argument('--models', nargs='+',
                        default=['claude-sonnet-4-5-20250929', 'gpt-4o-mini', 'claude-3-7-sonnet-20250219'],
                        help='Model IDs to evaluate')
    parser.add_argument('--dry-run', action='store_true',
                        help='Load questions but do not query models')
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()


    # Configure API keys
    print("Configuring API keys...")
    configure_api_keys(from_gcp=True)

    # Load questions
    print(f"Loading questions from {args.data_dir}...")
    data_dir = Path(args.data_dir)
    all_questions = load_questions(data_dir)
    print(f"Found {len(all_questions)} tech_comparative questions")

    if not all_questions:
        print("Error: No questions found")
        return 1

    # Subsample
    questions = subsample_questions(all_questions, args.num_questions, args.seed)
    print(f"Sampled {len(questions)} questions (seed={args.seed})")

    # Calculate base rate
    true_count = sum(1 for q in questions if q['ground_truth'])
    base_rate = true_count / len(questions)
    print(f"Sample base rate: {base_rate:.1%} ({true_count}/{len(questions)} true)")

    if args.dry_run:
        print("\n[Dry run - not querying models]")
        print("\nSample questions:")
        for q in questions[:5]:
            print(f"  - {q['question_text']}")
            print(f"    Answer: {q['ground_truth']}")
        return 0

    # Find models
    models_to_use = {}
    print(f"\nFinding models: {args.models}")
    for model_id in args.models:
        model = get_model_by_id(MODELS, model_id)
        if model:
            models_to_use[model_id] = model
            print(f"  Found: {model_id} -> {model.id}")
        else:
            print(f"  Warning: Model not found: {model_id}")

    if not models_to_use:
        print("Error: No valid models found")
        return 1

    # Prepare results structure
    results = {
        "metadata": {
            "seed": args.seed,
            "num_questions": len(questions),
            "timestamp": datetime.now().isoformat() + "Z",
            "base_rate": base_rate,
            "models": list(models_to_use.keys()),
        },
        "model_results": {},
        "questions": [],
    }

    # Process each question
    print(f"\nEvaluating {len(questions)} questions across {len(models_to_use)} models...")

    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {q['game_id']}/{q['question_id']}")
        print(f"  Q: {q['question_text'][:80]}...")
        print(f"  Ground truth: {q['ground_truth']}")

        # Load world report for this game
        world_report = load_world_report(data_dir, q['game_id'])
        if not world_report:
            print(f"  Warning: No world report found for {q['game_id']}")
            continue

        # Build prompt
        prompt = build_prompt(q['question_text'], world_report)

        # Query each model
        predictions = {}
        for model_id, model in models_to_use.items():
            print(f"  Querying {model_id}...", end=" ", flush=True)
            prob = query_model(model, prompt)
            if prob is not None:
                predictions[model_id] = prob
                print(f"{prob:.2f}")
            else:
                print("FAILED")

        # Store question result
        results["questions"].append({
            "game_id": q["game_id"],
            "question_id": q["question_id"],
            "question_text": q["question_text"],
            "ground_truth": q["ground_truth"],
            "predictions": predictions,
        })

    # Compute Brier scores for each model
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for model_id in models_to_use:
        predictions = []
        outcomes = []
        for qr in results["questions"]:
            if model_id in qr["predictions"]:
                predictions.append(qr["predictions"][model_id])
                outcomes.append(qr["ground_truth"])

        brier = compute_brier_score(predictions, outcomes)
        results["model_results"][model_id] = {
            "brier_score": brier,
            "num_predictions": len(predictions),
            "predictions": predictions,
        }

        print(f"\n{model_id}:")
        print(f"  Brier Score: {brier:.4f}")
        print(f"  Predictions: {len(predictions)}/{len(questions)}")

    # Reference: uninformed baseline
    uninformed_brier = compute_brier_score(
        [base_rate] * len(questions),
        [q["ground_truth"] for q in questions]
    )
    print(f"\nBaseline (always predict {base_rate:.2f}):")
    print(f"  Brier Score: {uninformed_brier:.4f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"data/evaluations/eval_{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
