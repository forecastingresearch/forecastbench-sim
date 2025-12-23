#!/usr/bin/env python3
"""
Parallel LLM evaluation for CivBench forecasting questions.

Features:
- Stratified sampling across composite difficulty levels (2-6)
- Parallel queries across models with per-provider rate limiting
- Checkpoint/resume for long-running evaluations
- Comprehensive logging for debugging

Usage:
    # Run with default models (ForecastBench + frontier)
    python scripts/evaluate_llm_forecasts_parallel.py --seed 42 --questions-per-difficulty 20

    # Dry run to inspect questions without querying models
    python scripts/evaluate_llm_forecasts_parallel.py --dry-run --questions-per-difficulty 5

    # Resume from checkpoint
    python scripts/evaluate_llm_forecasts_parallel.py --resume logs/eval_20251216_143000/checkpoint.json

    # Specify specific models
    python scripts/evaluate_llm_forecasts_parallel.py --models claude-3-7-sonnet-20250219 gpt-4o-mini

Requirements:
    pip install fri-utils python-dotenv
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.llm.model_registry import configure_api_keys, MODELS

from civrealm.evaluation.sampling import (
    load_all_questions,
    stratified_sample_batched,
    get_difficulty_distribution,
)
from civrealm.evaluation.rate_limiter import ProviderRateLimiter
from civrealm.evaluation.parallel_evaluator import run_batch_evaluation
from civrealm.metrics import compute_brier_score, compute_calibration_error

# - [X] claude forecastbench models
# - [ ] gpt/gemini fast models
# - [ ] claude frontier models
# - [ ] reasoning
# - [ ] merge evals/recompute summary statistics

# Models with ForecastBench scores for validation
FORECASTBENCH_MODELS = [
    # "claude-3-7-sonnet-20250219",
    # "claude-opus-4-1-20250805",
    # "claude-sonnet-4-20250514",
    "o3-2025-04-16",
    "gpt-4.1-2025-04-14",
    "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    # "DeepSeek-V3.1",
    # "Qwen3-235B-A22B-fp8-tput",
    # "Kimi-K2-Instruct",
    # "GLM-4.5-Air-FP8",
    # "mistral-large-2411",
]

# Frontier models without ForecastBench scores yet
FRONTIER_MODELS = [
    # "claude-opus-4-5-20251101",
    # "claude-sonnet-4-5-20250929",
    "gemini-3-pro-preview",
    "gpt-5.1-2025-11-13",
]


def setup_logging(run_id: str, log_dir: Path) -> logging.Logger:
    """Configure structured logging for the evaluation run."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("civbench_eval")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler (INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    ))

    # File handler (DEBUG)
    file_handler = logging.FileHandler(log_dir / "main.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    # Error-only handler
    error_handler = logging.FileHandler(log_dir / "errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger


def get_model_by_id(models_list, model_id: str):
    """Find a model in the MODELS list by ID (partial match)."""
    for model in models_list:
        if model_id in model.id or model.id in model_id:
            return model
    return None


def compute_metrics(results: list[dict], models: list) -> dict:
    """
    Compute evaluation metrics for each model.

    Returns dict with per-model metrics including:
    - brier_score: Overall Brier score
    - brier_by_difficulty: Brier score broken down by difficulty level
    - brier_by_template: Brier score broken down by template ID
    - brier_grid: Brier score by (template_id, horizon)
    - ece: Expected Calibration Error
    - num_predictions: Count of successful predictions
    - num_failures: Count of failed predictions
    """
    model_metrics = {}

    for model in models:
        model_id = model.id

        # Collect predictions and outcomes
        predictions = []
        outcomes = []
        by_difficulty: dict[int, tuple[list[float], list[bool]]] = defaultdict(
            lambda: ([], [])
        )
        by_template: dict[str, tuple[list[float], list[bool]]] = defaultdict(
            lambda: ([], [])
        )
        by_template_horizon: dict[tuple[str, str], tuple[list[float], list[bool]]] = defaultdict(
            lambda: ([], [])
        )

        for qr in results:
            pred_data = qr.get("predictions", {}).get(model_id, {})
            prob = pred_data.get("probability")

            if prob is not None:
                predictions.append(prob)
                outcomes.append(qr["ground_truth"])

                # Group by difficulty
                difficulty = qr.get("difficulty", {}).get("composite", 0)
                by_difficulty[difficulty][0].append(prob)
                by_difficulty[difficulty][1].append(qr["ground_truth"])
                # Group by template
                template_id = qr.get("template_id", "unknown")
                by_template[template_id][0].append(prob)
                by_template[template_id][1].append(qr["ground_truth"])
                # Group by template + horizon
                horizon = qr.get("difficulty", {}).get("horizon", "H1")
                by_template_horizon[(template_id, horizon)][0].append(prob)
                by_template_horizon[(template_id, horizon)][1].append(qr["ground_truth"])

        # Compute overall metrics
        brier = compute_brier_score(predictions, outcomes) if predictions else float('nan')
        ece = compute_calibration_error(predictions, outcomes) if predictions else float('nan')

        # Compute per-difficulty Brier scores
        brier_by_difficulty = {}
        for d, (preds, outs) in sorted(by_difficulty.items()):
            if preds:
                brier_by_difficulty[d] = compute_brier_score(preds, outs)

        # Compute per-template Brier scores
        brier_by_template = {}
        for t, (preds, outs) in sorted(by_template.items()):
            if preds:
                brier_by_template[t] = compute_brier_score(preds, outs)

        # Compute template x horizon grid of Brier scores
        brier_grid = {}
        for (t, h), (preds, outs) in sorted(by_template_horizon.items()):
            if preds:
                brier_grid.setdefault(t, {})[h] = compute_brier_score(preds, outs)

        # Count failures
        num_failures = sum(
            1 for qr in results
            if qr.get("predictions", {}).get(model_id, {}).get("probability") is None
        )

        model_metrics[model_id] = {
            "brier_score": brier,
            "brier_by_difficulty": brier_by_difficulty,
            "brier_by_template": brier_by_template,
            "brier_grid": brier_grid,
            "ece": ece,
            "num_predictions": len(predictions),
            "num_failures": num_failures,
        }

    return model_metrics


def print_results_summary(model_metrics: dict, base_rate: float, logger: logging.Logger):
    """Print a formatted summary of results."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    # Sort by Brier score
    sorted_models = sorted(
        model_metrics.items(),
        key=lambda x: x[1]["brier_score"] if not (x[1]["brier_score"] != x[1]["brier_score"]) else float('inf')
    )

    for model_id, metrics in sorted_models:
        logger.info(f"\n{model_id}:")
        logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
        logger.info(f"  ECE: {metrics['ece']:.4f}")
        logger.info(f"  Predictions: {metrics['num_predictions']} (failures: {metrics['num_failures']})")

        if metrics['brier_by_difficulty']:
            diff_str = ", ".join(
                f"d{d}={b:.3f}" for d, b in sorted(metrics['brier_by_difficulty'].items())
            )
            logger.info(f"  By difficulty: {diff_str}")

        if metrics['brier_by_template']:
            tmpl_str = ", ".join(
                f"{t}={b:.3f}" for t, b in sorted(metrics['brier_by_template'].items())
            )
            logger.info(f"  By template: {tmpl_str}")

        if metrics.get("brier_grid"):
            logger.info("  Template x Horizon (Brier):")
            for template_id in sorted(metrics["brier_grid"].keys()):
                row = metrics["brier_grid"][template_id]
                h1 = row.get("H1")
                h2 = row.get("H2")
                h3 = row.get("H3")
                h1_str = f"{h1:.3f}" if h1 is not None else "NA"
                h2_str = f"{h2:.3f}" if h2 is not None else "NA"
                h3_str = f"{h3:.3f}" if h3 is not None else "NA"
                logger.info(
                    f"    {template_id}: H1={h1_str} H2={h2_str} H3={h3_str}"
                )

    # Reference baselines
    uninformed_brier = base_rate * (1 - base_rate) + (1 - base_rate) * base_rate**2
    # Simpler: just use base_rate as constant prediction
    uninformed_brier = (1 - base_rate) * base_rate**2 + base_rate * (1 - base_rate)**2

    logger.info(f"\nBaseline (always predict {base_rate:.2f}):")
    logger.info(f"  Brier Score: {uninformed_brier:.4f}")


def compute_template_model_spread(results: list[dict], models: list) -> list[dict]:
    """Compute per-template model separation using Brier scores."""
    model_order = [m.id for m in models]
    template_spread = []
    for template_id in sorted({q.get("template_id", "unknown") for q in results}):
        by_model = {}
        for model in models:
            model_id = model.id
            preds = []
            outs = []
            for qr in results:
                if qr.get("template_id", "unknown") != template_id:
                    continue
                prob = qr.get("predictions", {}).get(model_id, {}).get("probability")
                if prob is not None:
                    preds.append(prob)
                    outs.append(qr["ground_truth"])
            if preds:
                by_model[model_id] = compute_brier_score(preds, outs)
        brier_values = list(by_model.values())
        if len(brier_values) >= 2:
            pairwise_diffs = []
            for i in range(len(brier_values)):
                for j in range(i + 1, len(brier_values)):
                    pairwise_diffs.append(abs(brier_values[i] - brier_values[j]))
            separation = sum(pairwise_diffs) / len(pairwise_diffs) if pairwise_diffs else float("nan")
        else:
            separation = float("nan")
        template_spread.append({
            "template_id": template_id,
            "brier_by_model": by_model,
            "model_order": model_order,
            "separation": separation,
        })
    return template_spread


def print_template_spread_summary(template_spread: list[dict], logger: logging.Logger):
    """Print per-template model spread summary."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEMPLATE DISCRIMINATION (Mean pairwise Brier distance)")
    logger.info("=" * 70)
    if template_spread:
        model_order = template_spread[0].get("model_order", [])
        if model_order:
            logger.info("Model order: " + ", ".join(model_order))
    ranked = sorted(
        template_spread,
        key=lambda x: x["separation"] if x["separation"] == x["separation"] else float("-inf"),
        reverse=True,
    )
    for entry in ranked:
        template_id = entry["template_id"]
        separation = entry["separation"]
        if separation != separation:
            logger.info(f"{template_id}: separation=NA (insufficient models)")
            continue
        model_order = entry.get("model_order", [])
        by_model = entry.get("brier_by_model", {})
        ordered_scores = []
        for model_id in model_order:
            if model_id in by_model:
                ordered_scores.append(f"({model_id}, {by_model[model_id]:.4f})")
        ordered_str = ", ".join(ordered_scores) if ordered_scores else "NA"
        logger.info(f"{template_id}: separation={separation:.4f} | {ordered_str}")


async def main():
    parser = argparse.ArgumentParser(
        description="Parallel LLM evaluation for CivBench forecasting questions"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/questions",
        help="Directory containing question data"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--questions-per-difficulty", "-n", type=int, default=None,
        help="Questions to sample per difficulty level (total = n * 5)"
    )
    parser.add_argument(
        "--min-per-template", type=int, default=0,
        help="Minimum questions to include per template_id (default: 0)"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=None,
        help="Model IDs to evaluate (default: ForecastBench + frontier models)"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=5,
        help="Save checkpoint every N completed batches (default: 5)"
    )
    parser.add_argument(
        "--resume", type=str,
        help="Resume from checkpoint file"
    )
    parser.add_argument(
        "--concurrent-batches", type=int, default=5,
        help="Number of game batches to process concurrently (default: 5)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load questions but do not query models"
    )
    parser.add_argument(
        "--forecastbench-only", action="store_true",
        help="Only evaluate models with ForecastBench scores"
    )
    parser.add_argument(
        "--timeout", type=int, default=180,
        help="Timeout per model query in seconds (default: 180). Use 0 for no timeout."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging: print full prompts and responses"
    )
    args = parser.parse_args()

    # Setup run ID and logging
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"eval_{run_id}"
    logger = setup_logging(run_id, log_dir)

    logger.info(f"CivBench Parallel Evaluation - Run {run_id}")
    logger.info(f"Log directory: {log_dir}")

    # Load environment variables
    load_dotenv()

    # Configure API keys (skip in dry-run mode)
    if not args.dry_run:
        logger.info("Configuring API keys...")
        configure_api_keys(
            from_gcp=True,
            # anthropic=os.getenv("ANTHROPIC_API_KEY"),
            # openai=os.getenv("OPENAI_API_KEY"),
            # google=os.getenv("GOOGLE_API_KEY"),
            # together=os.getenv("TOGETHER_API_KEY"),
            # mistral=os.getenv("MISTRAL_API_KEY"),
        )

    # Load all questions
    data_dir = Path(args.data_dir)
    logger.info(f"Loading questions from {data_dir}...")
    all_questions = load_all_questions(data_dir)
    logger.info(f"Found {len(all_questions)} total questions")

    if not all_questions:
        logger.error("No questions found")
        return 1

    # Show available difficulty distribution
    full_dist = get_difficulty_distribution(all_questions)
    logger.info(f"Full difficulty distribution: {full_dist}")

    # Stratified sampling by difficulty, then batched by game for token efficiency
    if args.questions_per_difficulty is None and args.min_per_template <= 0:
        logger.error("Must set --questions-per-difficulty or --min-per-template")
        return 1

    if args.questions_per_difficulty is None:
        logger.info(
            f"\nSampling minimum per template only (min per template: {args.min_per_template})..."
        )
    else:
        logger.info(
            f"\nSampling {args.questions_per_difficulty} questions per difficulty level "
            f"(min per template: {args.min_per_template})..."
        )
    question_batches = stratified_sample_batched(
        all_questions,
        per_difficulty=args.questions_per_difficulty,
        seed=args.seed,
        min_per_template=args.min_per_template if args.min_per_template > 0 else None,
    )
    # Flatten for metrics computation
    questions = [q for batch in question_batches for q in batch]
    logger.info(
        f"Sampled {len(questions)} questions across {len(question_batches)} game batches (seed={args.seed})"
    )

    # Show sampled distribution
    sampled_dist = get_difficulty_distribution(questions)
    logger.info(f"Sampled difficulty distribution: {sampled_dist}")

    # Calculate base rate
    true_count = sum(1 for q in questions if q['ground_truth'])
    base_rate = true_count / len(questions)
    logger.info(f"Sample base rate: {base_rate:.1%} ({true_count}/{len(questions)} true)")

    if args.dry_run:
        logger.info("\n[Dry run - not querying models]")
        logger.info(f"\nSample batches ({len(question_batches)} games):")
        for batch in question_batches[:3]:
            game_id = batch[0]["game_id"] if batch else "?"
            logger.info(f"\n  Game: {game_id}")
            for q in batch[:3]:
                logger.info(f"    - [{q['difficulty'].get('composite', '?')}] {q['question_text'][:60]}...")
            if len(batch) > 3:
                logger.info(f"    ... and {len(batch) - 3} more questions")
        if len(question_batches) > 3:
            logger.info(f"\n  ... and {len(question_batches) - 3} more game batches")
        return 0

    # Determine which models to use
    if args.models:
        model_ids = args.models
    elif args.forecastbench_only:
        model_ids = FORECASTBENCH_MODELS
    else:
        model_ids = FORECASTBENCH_MODELS + FRONTIER_MODELS

    # Find models in registry
    models_to_use = []
    logger.info(f"\nFinding models ({len(model_ids)} requested)...")
    for model_id in model_ids:
        model = get_model_by_id(MODELS, model_id)
        if model:
            models_to_use.append(model)
            logger.info(f"  Found: {model_id} -> {model.id}")
        else:
            logger.warning(f"  Not found: {model_id}")

    if not models_to_use:
        logger.error("No valid models found")
        return 1

    logger.info(f"\nWill evaluate {len(models_to_use)} models on {len(questions)} questions")

    # Prepare metadata
    metadata = {
        "run_id": run_id,
        "seed": args.seed,
        "questions_per_difficulty": args.questions_per_difficulty,
        "total_questions": len(questions),
        "base_rate": base_rate,
        "difficulty_distribution": sampled_dist,
        "models": [m.id for m in models_to_use],
        "start_time": datetime.now().isoformat() + "Z",
    }

    # Setup rate limiter
    rate_limiter = ProviderRateLimiter()

    # Determine checkpoint file
    if args.resume:
        checkpoint_file = Path(args.resume)
    else:
        checkpoint_file = log_dir / "checkpoint.json"

    # Run evaluation
    timeout = args.timeout if args.timeout > 0 else None
    timeout_str = f", timeout={timeout}s" if timeout else ", no timeout"
    start_time = datetime.now()

    logger.info(
        f"\nStarting evaluation ({len(question_batches)} game batches, "
        f"concurrency={args.concurrent_batches}{timeout_str})..."
    )
    results = await run_batch_evaluation(
        question_batches=question_batches,
        models=models_to_use,
        rate_limiter=rate_limiter,
        data_dir=data_dir,
        checkpoint_file=checkpoint_file,
        checkpoint_interval=args.checkpoint_interval,
        metadata=metadata,
        timeout=timeout,
        verbose=args.verbose,
        concurrent_batches=args.concurrent_batches,
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"\nEvaluation completed in {duration:.1f} seconds")

    # Compute metrics
    model_metrics = compute_metrics(results, models_to_use)
    template_spread = compute_template_model_spread(results, models_to_use)

    # Print summary
    print_results_summary(model_metrics, base_rate, logger)
    print_template_spread_summary(template_spread, logger)

    # Build final output
    metadata["end_time"] = end_time.isoformat() + "Z"
    metadata["duration_seconds"] = duration

    output = {
        "metadata": metadata,
        "model_results": model_metrics,
        "template_spread": template_spread,
        "questions": results,
    }

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"data/evaluations/parallel_eval_{args.seed}_{run_id}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
