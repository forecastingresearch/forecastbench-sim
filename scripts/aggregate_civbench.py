#!/usr/bin/env python3
"""
Aggregate CivBench results from JSON files and compute bootstrapped confidence intervals.

Reads all evaluation JSON files from data/evaluations/, filters out h0 (horizon-0) questions,
and computes Brier scores with 95% CIs via bootstrapping.
"""

import json
import glob
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_evaluation_json(filepath: str) -> dict:
    """Load a single evaluation JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_predictions(data: dict, filepath: str) -> list[dict]:
    """
    Extract question-level predictions from evaluation data.

    Returns list of dicts with: question_id, model_id, probability, ground_truth, template_id
    """
    predictions = []
    run_id = data.get('metadata', {}).get('run_id', os.path.basename(filepath))

    for question in data.get('questions', []):
        question_id = question.get('question_id')
        template_id = question.get('template_id', '')
        ground_truth = question.get('ground_truth')

        if ground_truth is None:
            continue

        for model_id, pred_data in question.get('predictions', {}).items():
            prob = pred_data.get('probability')
            if prob is not None:
                predictions.append({
                    'question_id': question_id,
                    'model_id': model_id,
                    'probability': prob,
                    'ground_truth': 1.0 if ground_truth else 0.0,
                    'template_id': template_id,
                    'run_id': run_id,
                    'filepath': filepath
                })

    return predictions


def filter_non_h0(predictions: list[dict]) -> list[dict]:
    """Filter out h0 (horizon-0) questions - these are trivial current-state questions."""
    return [p for p in predictions if not p['template_id'].startswith('h0_')]


def deduplicate_predictions(predictions: list[dict]) -> list[dict]:
    """
    Deduplicate predictions - if same (question_id, model_id) appears multiple times,
    keep the one from the latest run_id.
    """
    # Group by (question_id, model_id)
    grouped = defaultdict(list)
    for p in predictions:
        key = (p['question_id'], p['model_id'])
        grouped[key].append(p)

    # Keep the one with the latest run_id
    deduped = []
    for key, preds in grouped.items():
        # Sort by run_id descending and take first
        latest = sorted(preds, key=lambda x: x['run_id'], reverse=True)[0]
        deduped.append(latest)

    return deduped


def compute_brier_score(probs: np.ndarray, truths: np.ndarray) -> float:
    """Compute Brier score: mean squared error between probability and outcome."""
    return np.mean((probs - truths) ** 2)


def bootstrap_ci(probs: np.ndarray, truths: np.ndarray,
                 n_bootstrap: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    """
    Compute confidence interval for Brier score via bootstrapping.

    Returns (ci_lower, ci_upper)
    """
    n = len(probs)
    rng = np.random.default_rng(42)  # For reproducibility

    bootstrap_scores = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.integers(0, n, size=n)
        bs_probs = probs[indices]
        bs_truths = truths[indices]
        bs_score = compute_brier_score(bs_probs, bs_truths)
        bootstrap_scores.append(bs_score)

    bootstrap_scores = np.array(bootstrap_scores)
    alpha = (1 - ci) / 2
    ci_lower = np.percentile(bootstrap_scores, alpha * 100)
    ci_upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)

    return ci_lower, ci_upper


def aggregate_civbench(eval_dir: str, output_path: str, n_bootstrap: int = 10000):
    """
    Main function to aggregate CivBench results.

    Args:
        eval_dir: Directory containing evaluation JSON files
        output_path: Path to output CSV file
        n_bootstrap: Number of bootstrap iterations
    """
    # Find all JSON files
    json_pattern = os.path.join(eval_dir, 'parallel_eval_*.json')
    json_files = glob.glob(json_pattern)

    if not json_files:
        print(f"No evaluation files found matching {json_pattern}")
        return

    print(f"Found {len(json_files)} evaluation files")

    # Load all predictions
    all_predictions = []
    for filepath in sorted(json_files):
        print(f"  Loading {os.path.basename(filepath)}...")
        data = load_evaluation_json(filepath)
        preds = extract_predictions(data, filepath)
        all_predictions.extend(preds)

    print(f"Total predictions loaded: {len(all_predictions)}")

    # Filter out h0 questions
    non_h0_predictions = filter_non_h0(all_predictions)
    print(f"Non-h0 predictions: {len(non_h0_predictions)}")

    # Deduplicate
    deduped = deduplicate_predictions(non_h0_predictions)
    print(f"After deduplication: {len(deduped)}")

    # Group by model
    model_predictions = defaultdict(list)
    for p in deduped:
        model_predictions[p['model_id']].append(p)

    print(f"\nFound {len(model_predictions)} unique models")

    # Compute Brier score and CI for each model
    results = []
    for model_id in sorted(model_predictions.keys()):
        preds = model_predictions[model_id]
        probs = np.array([p['probability'] for p in preds])
        truths = np.array([p['ground_truth'] for p in preds])

        brier = compute_brier_score(probs, truths)
        ci_lower, ci_upper = bootstrap_ci(probs, truths, n_bootstrap=n_bootstrap)

        results.append({
            'model': model_id,
            'brier_score': brier,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_questions': len(preds)
        })

        print(f"  {model_id}: {brier:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] (n={len(preds)})")

    # Save to CSV
    df = pd.DataFrame(results)
    df = df.sort_values('brier_score')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Aggregate CivBench results with bootstrapped CIs')
    parser.add_argument('--eval-dir', default='data/evaluations',
                        help='Directory containing evaluation JSON files')
    parser.add_argument('--output', default='data/civbench_results.csv',
                        help='Output CSV path')
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                        help='Number of bootstrap iterations')

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    eval_dir = args.eval_dir
    if not os.path.isabs(eval_dir):
        eval_dir = project_root / eval_dir

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = project_root / output_path

    aggregate_civbench(str(eval_dir), str(output_path), args.n_bootstrap)
