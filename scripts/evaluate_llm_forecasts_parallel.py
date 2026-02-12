#!/usr/bin/env python3
"""
Parallel LLM evaluation for CivBench forecasting questions.

Features:
- Stratified sampling across question templates
- Parallel queries across models with per-provider rate limiting
- Checkpoint/resume for long-running evaluations
- Comprehensive logging for debugging

Usage:
    # Run with default models (ForecastBench + frontier)
    python scripts/evaluate_llm_forecasts_parallel.py --seed 42 --questions-per-template 10

    # Dry run to inspect questions without querying models
    python scripts/evaluate_llm_forecasts_parallel.py --dry-run --questions-per-template 5

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

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from civrealm.evaluation.models import get_models, load_api_keys_from_gcp

# Load API keys from GCP Secret Manager (uses GOOGLE_CLOUD_PROJECT from .env)
load_api_keys_from_gcp()
from civrealm.evaluation.sampling import (
    load_all_questions,
    stratified_sample_batched,
    stratified_sample_by_horizon_template,
    get_template_distribution,
    get_horizon_template_distribution,
)
from civrealm.evaluation.rate_limiter import ProviderRateLimiter
from civrealm.evaluation.parallel_evaluator import run_batch_evaluation, run_continuous_batch_evaluation
from civrealm.metrics import compute_brier_score, compute_calibration_error, compute_crps, compute_aggregate_crps, compute_aggregate_mae

# - [X] claude forecastbench models
# - [ ] gpt/gemini fast models
# - [ ] claude frontier models
# - [ ] reasoning
# - [ ] merge evals/recompute summary statistics

# Models with ForecastBench scores for validation (LiteLLM format: provider/model)
FORECASTBENCH_MODELS = [
    # "anthropic/claude-3-7-sonnet-20250219",
    # "anthropic/claude-opus-4-1-20250805",
    # "anthropic/claude-sonnet-4-20250514",
    "openai/o3-2025-04-16",
    "openai/gpt-4.1-2025-04-14",
    "openai/gpt-5-2025-08-07",
    "openai/gpt-5-mini-2025-08-07",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    # "together/DeepSeek-V3.1",
    # "together/Qwen3-235B-A22B-fp8-tput",
    # "together/Kimi-K2-Instruct",
    # "together/GLM-4.5-Air-FP8",
    # "mistral/mistral-large-2411",
]

# Frontier models without ForecastBench scores yet (LiteLLM format: provider/model)
FRONTIER_MODELS = [
    "anthropic/claude-opus-4-5-20251101",
    "anthropic/claude-sonnet-4-5-20250929",
    "google/gemini-3-pro-preview",
    "openai/gpt-5.1-2025-11-13",
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




def _compute_binary_metrics(results: list[dict], model_id: str) -> dict:
    """Compute metrics for binary questions."""
    # Collect predictions and outcomes
    predictions = []
    outcomes = []
    by_template: dict[str, tuple[list[float], list[bool]]] = defaultdict(
        lambda: ([], [])
    )

    for qr in results:
        pred_data = qr.get("predictions", {}).get(model_id, {})
        prob = pred_data.get("probability")

        if prob is not None:
            predictions.append(prob)
            outcomes.append(qr["ground_truth"])

            # Group by template
            template_id = qr.get("template_id", "unknown")
            by_template[template_id][0].append(prob)
            by_template[template_id][1].append(qr["ground_truth"])

    # Compute overall metrics
    brier = compute_brier_score(predictions, outcomes) if predictions else float('nan')
    ece = compute_calibration_error(predictions, outcomes) if predictions else float('nan')

    # Compute per-template Brier scores
    brier_by_template = {}
    for t, (preds, outs) in sorted(by_template.items()):
        if preds:
            brier_by_template[t] = compute_brier_score(preds, outs)

    # Count failures
    num_failures = sum(
        1 for qr in results
        if qr.get("predictions", {}).get(model_id, {}).get("probability") is None
    )

    return {
        "brier_score": brier,
        "brier_by_template": brier_by_template,
        "ece": ece,
        "num_predictions": len(predictions),
        "num_failures": num_failures,
    }


def _compute_continuous_metrics(results: list[dict], model_id: str) -> dict:
    """Compute metrics for continuous questions."""
    all_percentiles = []
    all_true_values = []
    by_template = defaultdict(lambda: ([], []))

    for qr in results:
        pred_data = qr.get("predictions", {}).get(model_id, {})
        percentiles = pred_data.get("percentiles")
        if percentiles and all(k in percentiles for k in ["p10", "p25", "p50", "p75", "p90"]):
            all_percentiles.append(percentiles)
            all_true_values.append(qr["ground_truth"])
            template_id = qr.get("template_id", "unknown")
            by_template[template_id][0].append(percentiles)
            by_template[template_id][1].append(qr["ground_truth"])

    if not all_percentiles:
        return {
            "crps": float('nan'),
            "mae": float('nan'),
            "crps_by_template": {},
            "mae_by_template": {},
            "num_predictions": 0,
            "num_failures": len(results),
        }

    crps = compute_aggregate_crps(all_percentiles, all_true_values)
    mae = compute_aggregate_mae(
        [p["p50"] for p in all_percentiles],
        all_true_values
    )

    crps_by_template = {}
    mae_by_template = {}
    for t, (pcts, vals) in sorted(by_template.items()):
        crps_by_template[t] = compute_aggregate_crps(pcts, vals)
        mae_by_template[t] = compute_aggregate_mae([p["p50"] for p in pcts], vals)

    return {
        "crps": crps,
        "mae": mae,
        "crps_by_template": crps_by_template,
        "mae_by_template": mae_by_template,
        "num_predictions": len(all_percentiles),
        "num_failures": len(results) - len(all_percentiles),
    }


def compute_metrics(results: list[dict], models: list) -> dict:
    """
    Compute evaluation metrics for each model.

    Returns dict with per-model metrics including binary and continuous sections.
    """
    model_metrics = {}

    for model in models:
        model_id = model.id

        # Split results by type
        binary_results = [r for r in results if r.get("question_type", "binary") == "binary"]
        continuous_results = [r for r in results if r.get("question_type") == "continuous"]

        # Binary metrics (existing logic)
        binary_metrics = _compute_binary_metrics(binary_results, model_id)

        # Continuous metrics (new)
        continuous_metrics = _compute_continuous_metrics(continuous_results, model_id)

        model_metrics[model_id] = {
            "binary": binary_metrics,
            "continuous": continuous_metrics,
        }

    return model_metrics


def print_results_summary(model_metrics: dict, base_rate: float, logger: logging.Logger):
    """Print a formatted summary of results."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    for model_id, metrics in sorted(model_metrics.items()):
        logger.info(f"\n{model_id}:")

        # Binary section
        binary = metrics.get("binary", {})
        if binary.get("num_predictions", 0) > 0:
            logger.info(f"  Binary:")
            logger.info(f"    Brier Score: {binary['brier_score']:.4f}")
            logger.info(f"    ECE: {binary['ece']:.4f}")
            logger.info(f"    Predictions: {binary['num_predictions']} (failures: {binary['num_failures']})")
            if binary.get('brier_by_template'):
                logger.info("    By template:")
                for t, b in sorted(binary['brier_by_template'].items()):
                    logger.info(f"      {t}: {b:.4f}")

        # Continuous section
        continuous = metrics.get("continuous", {})
        if continuous.get("num_predictions", 0) > 0:
            logger.info(f"  Continuous:")
            logger.info(f"    CRPS: {continuous['crps']:.2f}")
            logger.info(f"    MAE: {continuous['mae']:.2f}")
            logger.info(f"    Predictions: {continuous['num_predictions']} (failures: {continuous['num_failures']})")
            if continuous.get('crps_by_template'):
                logger.info("    By template (CRPS / MAE):")
                for t in sorted(continuous['crps_by_template'].keys()):
                    c = continuous['crps_by_template'][t]
                    m = continuous.get('mae_by_template', {}).get(t, float('nan'))
                    logger.info(f"      {t}: {c:.2f} / {m:.2f}")

    # Reference baselines (binary only)
    if base_rate > 0:
        uninformed_brier = (1 - base_rate) * base_rate**2 + base_rate * (1 - base_rate)**2
        logger.info(f"\nBinary Baseline (always predict {base_rate:.2f}):")
        logger.info(f"  Brier Score: {uninformed_brier:.4f}")


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
        "--questions-per-template", "-n", type=int, default=10,
        help="Questions to sample per template (total = n * num_templates, default: 10)"
    )
    parser.add_argument(
        "--questions-per-horizon-template", type=int, default=None,
        help="Questions to sample per (horizon, template) pair. Overrides --questions-per-template. "
             "Use 1 for anchor runs (e.g., 13 templates * 8 horizons = up to 104 questions)"
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
    parser.add_argument(
        "--horizon", type=str, nargs="+",
        choices=["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"],
        help="Filter questions by horizon (H0=comprehension, H1=30 turns, H2=60 turns, H3=90 turns, H4=120 turns, H5=150 turns, H6=180 turns, H7=210 turns)"
    )
    parser.add_argument(
        "--min-difficulty", type=float,
        help="Minimum difficulty score (0.0-1.0, higher = harder)"
    )
    parser.add_argument(
        "--max-difficulty", type=float,
        help="Maximum difficulty score (0.0-1.0, higher = harder)"
    )
    parser.add_argument(
        "--update-difficulty", action="store_true",
        help="Recompute and update difficulty scores after evaluation completes"
    )
    parser.add_argument(
        "--question-type", type=str, default="all",
        choices=["all", "binary", "continuous"],
        help="Filter questions by type (default: all)"
    )
    args = parser.parse_args()

    # Setup run ID and logging
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"eval_{run_id}"
    logger = setup_logging(run_id, log_dir)

    logger.info(f"CivBench Parallel Evaluation - Run {run_id}")
    logger.info(f"Log directory: {log_dir}")

    # Load all questions (with optional difficulty filtering)
    data_dir = Path(args.data_dir)
    logger.info(f"Loading questions from {data_dir}...")
    all_questions = load_all_questions(
        data_dir,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
    )
    logger.info(f"Found {len(all_questions)} total questions")

    if args.min_difficulty is not None or args.max_difficulty is not None:
        logger.info(f"Difficulty filter: min={args.min_difficulty}, max={args.max_difficulty}")

    if not all_questions:
        logger.error("No questions found")
        return 1

    # Filter by horizon if specified
    if args.horizon:
        horizon_set = set(args.horizon)
        all_questions = [q for q in all_questions if q.get("horizon") in horizon_set]
        logger.info(f"Filtered to {len(all_questions)} questions with horizons: {args.horizon}")

        if not all_questions:
            logger.error(f"No questions found with horizons: {args.horizon}")
            return 1

    # Filter by question type if specified
    if args.question_type != "all":
        all_questions = [q for q in all_questions if q.get("question_type", "binary") == args.question_type]
        logger.info(f"Filtered to {len(all_questions)} {args.question_type} questions")

    # Show available template distribution
    full_dist = get_template_distribution(all_questions)
    logger.info(f"Full template distribution: {full_dist}")

    # Stratified sampling - either by (horizon, template) pair or by template only
    if args.questions_per_horizon_template is not None:
        # Fine-grained stratification by (horizon, template) pair
        logger.info(f"\nSampling {args.questions_per_horizon_template} questions per (horizon, template) pair...")
        questions = stratified_sample_by_horizon_template(
            all_questions,
            per_pair=args.questions_per_horizon_template,
            seed=args.seed,
            horizons=args.horizon,  # Use horizon filter if specified
        )
        sampling_note = f"(horizon-template stratified, seed={args.seed})"
    else:
        # Standard stratification by template only
        logger.info(f"\nSampling {args.questions_per_template} questions per template...")
        question_batches = stratified_sample_batched(
            all_questions,
            per_template=args.questions_per_template,
            seed=args.seed,
        )
        # Flatten for metrics computation
        questions = [q for batch in question_batches for q in batch]
        sampling_note = f"(seed={args.seed})"

    if not questions:
        logger.error("No questions sampled; check filters and sampling settings.")
        return 1

    # Group by game for batching, then chunk each game into fixed-size batches
    batch_size = 20
    by_game: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_game[q["game_id"]].append(q)

    question_batches = []
    for gid in sorted(by_game.keys()):
        game_questions = by_game[gid]
        for start in range(0, len(game_questions), batch_size):
            question_batches.append(game_questions[start:start + batch_size])
    logger.info(
        f"Sampled {len(questions)} questions across {len(question_batches)} batches "
        f"(chunk size={batch_size}) {sampling_note}"
    )

    # Split batches by question type for separate prompting
    binary_batches = []
    continuous_batches = []
    for batch in question_batches:
        binary_qs = [q for q in batch if q.get("question_type", "binary") == "binary"]
        continuous_qs = [q for q in batch if q.get("question_type") == "continuous"]
        if binary_qs:
            binary_batches.append(binary_qs)
        if continuous_qs:
            continuous_batches.append(continuous_qs)

    logger.info(f"Split into {len(binary_batches)} binary batches, {len(continuous_batches)} continuous batches")

    # Show sampled distribution
    sampled_dist = get_template_distribution(questions)
    logger.info(f"Sampled template distribution: {sampled_dist}")

    # Show horizon-template distribution when using that sampling mode
    if args.questions_per_horizon_template is not None:
        ht_dist = get_horizon_template_distribution(questions)
        logger.info("Sampled (horizon, template) distribution:")
        for horizon, templates in ht_dist.items():
            logger.info(f"  {horizon}: {len(templates)} templates, {sum(templates.values())} questions")

    # Calculate base rate (binary questions only)
    binary_questions = [q for q in questions if q.get("question_type", "binary") == "binary"]
    true_count = sum(1 for q in binary_questions if q['ground_truth']) if binary_questions else 0
    base_rate = true_count / len(binary_questions) if binary_questions else 0.5
    logger.info(f"Sample base rate (binary): {base_rate:.1%} ({true_count}/{len(binary_questions)} true)")

    # Track question type distribution
    type_dist = defaultdict(int)
    for q in questions:
        type_dist[q.get("question_type", "binary")] += 1
    logger.info(f"Question type distribution: {dict(type_dist)}")

    if args.dry_run:
        logger.info("\n[Dry run - not querying models]")
        logger.info(f"\nSample binary batches ({len(binary_batches)} total):")
        for batch in binary_batches[:2]:
            game_id = batch[0]["game_id"] if batch else "?"
            logger.info(f"\n  Game: {game_id}")
            for q in batch[:3]:
                logger.info(f"    - [{q.get('template_id', '?')}] {q['question_text'][:60]}...")
            if len(batch) > 3:
                logger.info(f"    ... and {len(batch) - 3} more questions")
        if len(binary_batches) > 2:
            logger.info(f"\n  ... and {len(binary_batches) - 2} more binary batches")

        logger.info(f"\nSample continuous batches ({len(continuous_batches)} total):")
        for batch in continuous_batches[:2]:
            game_id = batch[0]["game_id"] if batch else "?"
            logger.info(f"\n  Game: {game_id}")
            for q in batch[:3]:
                logger.info(f"    - [{q.get('template_id', '?')}] {q['question_text'][:60]}...")
            if len(batch) > 3:
                logger.info(f"    ... and {len(batch) - 3} more questions")
        if len(continuous_batches) > 2:
            logger.info(f"\n  ... and {len(continuous_batches) - 2} more continuous batches")
        return 0

    # Determine which models to use
    if args.models:
        model_ids = args.models
    elif args.forecastbench_only:
        model_ids = FORECASTBENCH_MODELS
    else:
        model_ids = FORECASTBENCH_MODELS + FRONTIER_MODELS

    # Create model objects from IDs
    models_to_use = get_models(model_ids)
    logger.info(f"\nCreated {len(models_to_use)} models: {[m.id for m in models_to_use]}")

    logger.info(f"\nWill evaluate {len(models_to_use)} models on {len(questions)} questions")

    # Prepare metadata
    metadata = {
        "run_id": run_id,
        "seed": args.seed,
        "questions_per_template": args.questions_per_template,
        "questions_per_horizon_template": args.questions_per_horizon_template,
        "sampling_mode": "horizon_template" if args.questions_per_horizon_template else "template",
        "total_questions": len(questions),
        "base_rate": base_rate,
        "question_type_distribution": dict(type_dist),
        "template_distribution": sampled_dist,
        "horizon_filter": args.horizon,
        "question_type_filter": args.question_type,
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

    # Run binary and continuous evaluations separately
    binary_results = []
    continuous_results = []

    if binary_batches:
        logger.info(
            f"\nStarting binary evaluation ({len(binary_batches)} game batches, "
            f"concurrency={args.concurrent_batches}{timeout_str})..."
        )
        binary_results = await run_batch_evaluation(
            question_batches=binary_batches,
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

    if continuous_batches:
        logger.info(
            f"\nStarting continuous evaluation ({len(continuous_batches)} game batches, "
            f"concurrency={args.concurrent_batches}{timeout_str})..."
        )
        continuous_results = await run_continuous_batch_evaluation(
            question_batches=continuous_batches,
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

    # Merge results
    results = binary_results + continuous_results

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"\nEvaluation completed in {duration:.1f} seconds")

    # Compute metrics
    model_metrics = compute_metrics(results, models_to_use)

    # Print summary
    print_results_summary(model_metrics, base_rate, logger)

    # Build final output
    metadata["end_time"] = end_time.isoformat() + "Z"
    metadata["duration_seconds"] = duration

    output = {
        "metadata": metadata,
        "model_results": model_metrics,
        "questions": results,
    }

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"data/evaluations/runs/parallel_eval_{args.seed}_{run_id}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    # Optionally update difficulty scores
    if args.update_difficulty:
        logger.info("\nUpdating difficulty scores...")
        try:
            from civrealm.evaluation.difficulty import (
                aggregate_evaluations,
                compute_question_difficulties,
                compute_percentile_ranks,
                update_question_files,
            )

            eval_dir = Path("data/evaluations/results")
            questions_dir = Path("data/questions")

            aggregated = aggregate_evaluations(eval_dir)
            difficulties = compute_question_difficulties(aggregated)
            difficulties = compute_percentile_ranks(difficulties)
            updated, files = update_question_files(questions_dir, difficulties)

            logger.info(f"Updated {updated} questions in {files} files")
        except Exception as e:
            logger.warning(f"Failed to update difficulty scores: {e}")

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
