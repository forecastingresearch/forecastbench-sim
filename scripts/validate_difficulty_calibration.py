#!/usr/bin/env python3
"""
Validate IRT difficulty calibration using held-out models.

This script validates that the difficulty scores computed from calibration
models generalize to held-out validation models by:
1. Computing Pearson correlation between difficulty and validation Brier scores
2. Generating box plots of Brier scores by difficulty tertile
3. Generating scatter plots of difficulty vs Brier with regression line

Usage:
    python scripts/validate_difficulty_calibration.py \
        --validation-models openai/gpt-5-mini google/gemini-2.5-flash \
        --questions-per-tertile 30 \
        --seed 42
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
from typing import Any

import numpy as np
from scipy import stats

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.llm.litellm_models import get_models, configure_api_keys
from civrealm.evaluation.parallel_evaluator import (
    evaluate_question_batch,
    ProviderRateLimiter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_questions_with_difficulty(questions_file: Path) -> list[dict]:
    """Load questions that have empirical difficulty scores."""
    with open(questions_file) as f:
        data = json.load(f)

    questions = []
    for q in data.get("questions", []):
        if "empirical_difficulty" in q and q["empirical_difficulty"].get("score") is not None:
            questions.append(q)

    return questions


def compute_tertiles(questions: list[dict]) -> dict[str, tuple[float, float]]:
    """
    Compute difficulty tertile boundaries.

    Returns:
        Dict with 'easy', 'medium', 'hard' keys, each mapping to (min, max) score range
    """
    scores = [q["empirical_difficulty"]["score"] for q in questions]
    p33 = np.percentile(scores, 33.33)
    p67 = np.percentile(scores, 66.67)

    return {
        "easy": (0.0, p33),
        "medium": (p33, p67),
        "hard": (p67, 1.0),
    }


def assign_tertile(score: float, bounds: dict[str, tuple[float, float]]) -> str:
    """Assign a difficulty score to a tertile."""
    if score <= bounds["easy"][1]:
        return "easy"
    elif score <= bounds["medium"][1]:
        return "medium"
    else:
        return "hard"


def sample_by_tertile(
    questions: list[dict],
    bounds: dict[str, tuple[float, float]],
    per_tertile: int,
    seed: int,
) -> dict[str, list[dict]]:
    """
    Sample questions stratified by difficulty tertile.

    Args:
        questions: Questions with empirical_difficulty
        bounds: Tertile boundaries from compute_tertiles()
        per_tertile: Number of questions to sample per tertile
        seed: Random seed

    Returns:
        Dict mapping tertile name to list of sampled questions
    """
    rng = np.random.default_rng(seed)

    # Group by tertile
    by_tertile: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        score = q["empirical_difficulty"]["score"]
        tertile = assign_tertile(score, bounds)
        by_tertile[tertile].append(q)

    # Sample from each tertile
    sampled: dict[str, list[dict]] = {}
    for tertile in ["easy", "medium", "hard"]:
        available = by_tertile[tertile]
        n = min(per_tertile, len(available))
        indices = rng.choice(len(available), size=n, replace=False)
        sampled[tertile] = [available[i] for i in indices]
        logger.info(f"  {tertile.capitalize()}: {len(sampled[tertile])} questions (from {len(available)} available)")

    return sampled


def compute_validation_metrics(
    results: list[dict],
    tertile_bounds: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """
    Compute validation metrics from evaluation results.

    Returns question-level data and summary statistics:
    - questions: List of per-question data (difficulty, brier, tertile)
    - pearson_r: Pearson correlation between difficulty and Brier
    - pearson_p: P-value for Pearson correlation
    - per_tertile: Summary stats by tertile

    Args:
        results: List of evaluation results with predictions
        tertile_bounds: From compute_tertiles()

    Returns:
        Dict of validation metrics including question-level data
    """
    # Extract question-level data
    question_data = []

    for r in results:
        if "empirical_difficulty" not in r:
            continue

        difficulty = r["empirical_difficulty"]["score"]
        ground_truth = r["ground_truth"]
        question_id = r.get("question_id")
        template_id = r.get("template_id")

        # Average Brier across all validation models
        model_briers = {}
        for model_id, pred in r.get("predictions", {}).items():
            prob = pred.get("probability")
            if prob is not None:
                brier = (prob - int(ground_truth)) ** 2
                model_briers[model_id] = brier

        if model_briers:
            avg_brier = np.mean(list(model_briers.values()))
            tertile = assign_tertile(difficulty, tertile_bounds)

            question_data.append({
                "question_id": question_id,
                "template_id": template_id,
                "difficulty": difficulty,
                "brier": avg_brier,
                "tertile": tertile,
                "ground_truth": ground_truth,
                "model_briers": model_briers,
            })

    if len(question_data) < 3:
        return {"error": "Insufficient data points", "questions": []}

    # Extract arrays for correlation
    difficulties = [q["difficulty"] for q in question_data]
    brier_scores = [q["brier"] for q in question_data]

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(difficulties, brier_scores)

    # Per-tertile stats
    tertile_briers: dict[str, list[float]] = defaultdict(list)
    for q in question_data:
        tertile_briers[q["tertile"]].append(q["brier"])

    per_tertile = {
        t: {
            "mean": float(np.mean(briers)) if briers else None,
            "std": float(np.std(briers)) if briers else None,
            "median": float(np.median(briers)) if briers else None,
            "min": float(np.min(briers)) if briers else None,
            "max": float(np.max(briers)) if briers else None,
            "n": len(briers),
            "brier_scores": [float(b) for b in briers],  # Raw data for box plots
        }
        for t, briers in tertile_briers.items()
    }

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "per_tertile": per_tertile,
        "n_questions": len(question_data),
        "questions": question_data,  # Question-level data
    }


def generate_visualizations(
    metrics: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """
    Generate box plot and scatter plot visualizations.

    Args:
        metrics: Output from compute_validation_metrics()
        output_dir: Directory to save plots

    Returns:
        Dict mapping plot name to file path
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping visualizations")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {}

    questions = metrics.get("questions", [])
    if not questions:
        return plots

    # --- Box Plot: Brier by Tertile ---
    fig, ax = plt.subplots(figsize=(8, 6))

    tertile_order = ["easy", "medium", "hard"]
    box_data = []
    labels = []
    for t in tertile_order:
        tertile_stats = metrics.get("per_tertile", {}).get(t, {})
        briers = tertile_stats.get("brier_scores", [])
        if briers:
            box_data.append(briers)
            labels.append(f"{t.capitalize()}\n(n={len(briers)})")

    if box_data:
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)

        # Color the boxes
        colors = ["#90EE90", "#FFD700", "#FF6B6B"]  # green, yellow, red
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel("Difficulty Tertile", fontsize=12)
        ax.set_ylabel("Brier Score", fontsize=12)
        ax.set_title("Validation Brier Scores by Difficulty Tertile", fontsize=14)
        ax.grid(axis="y", alpha=0.3)

        box_plot_path = output_dir / "brier_by_tertile_boxplot.png"
        plt.tight_layout()
        plt.savefig(box_plot_path, dpi=150)
        plt.close()
        plots["boxplot"] = box_plot_path
        logger.info(f"Saved box plot: {box_plot_path}")

    # --- Scatter Plot: Difficulty vs Brier ---
    fig, ax = plt.subplots(figsize=(10, 6))

    difficulties = [q["difficulty"] for q in questions]
    briers = [q["brier"] for q in questions]
    tertiles = [q["tertile"] for q in questions]

    # Color by tertile
    color_map = {"easy": "#90EE90", "medium": "#FFD700", "hard": "#FF6B6B"}
    colors = [color_map.get(t, "gray") for t in tertiles]

    ax.scatter(difficulties, briers, c=colors, alpha=0.6, edgecolors="black", linewidth=0.5)

    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(difficulties, briers)
    x_line = np.array([min(difficulties), max(difficulties)])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "r--", linewidth=2, label=f"r = {metrics['pearson_r']:.3f}, p = {metrics['pearson_p']:.4f}")

    ax.set_xlabel("Difficulty Score (from calibration models)", fontsize=12)
    ax.set_ylabel("Brier Score (validation models)", fontsize=12)
    ax.set_title("Difficulty vs Validation Brier Score", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # Add legend for tertiles
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#90EE90", edgecolor="black", label="Easy"),
        Patch(facecolor="#FFD700", edgecolor="black", label="Medium"),
        Patch(facecolor="#FF6B6B", edgecolor="black", label="Hard"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", title="Tertile")

    # Add correlation annotation
    ax.annotate(
        f"Pearson r = {metrics['pearson_r']:.3f}\np = {metrics['pearson_p']:.4f}",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    scatter_plot_path = output_dir / "difficulty_vs_brier_scatter.png"
    plt.tight_layout()
    plt.savefig(scatter_plot_path, dpi=150)
    plt.close()
    plots["scatter"] = scatter_plot_path
    logger.info(f"Saved scatter plot: {scatter_plot_path}")

    return plots


async def run_validation(
    questions: list[dict],
    models: list,
    rate_limiter: ProviderRateLimiter,
    world_reports_dir: Path,
    timeout: float,
) -> list[dict]:
    """
    Run validation evaluations on sampled questions.

    Args:
        questions: List of questions to evaluate
        models: List of LiteLLMModel objects
        rate_limiter: Rate limiter for API calls
        world_reports_dir: Directory containing world reports
        timeout: Timeout per model query

    Returns:
        List of evaluation results
    """
    # Group questions by game
    by_game: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        game_id = q.get("parameters", {}).get("game_id")
        if game_id:
            by_game[game_id].append(q)

    all_results = []

    for game_id, game_questions in by_game.items():
        # Load world report
        report_file = world_reports_dir / game_id / "world_report" / "turn_060_report.txt"
        if not report_file.exists():
            logger.warning(f"World report not found: {report_file}")
            continue

        with open(report_file) as f:
            world_report = f.read()

        logger.info(f"Evaluating {len(game_questions)} questions for game {game_id}")

        # Add required fields for evaluation
        for q in game_questions:
            q["game_id"] = game_id
            q["ground_truth"] = q.get("resolution", {}).get("answer")

        # Run evaluation
        results = await evaluate_question_batch(
            questions=game_questions,
            models=models,
            rate_limiter=rate_limiter,
            world_report=world_report,
            timeout=timeout,
        )

        # Merge difficulty info back
        for r, q in zip(results, game_questions):
            r["empirical_difficulty"] = q.get("empirical_difficulty")

        all_results.extend(results)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Validate IRT difficulty calibration using held-out models"
    )
    parser.add_argument(
        "--validation-models",
        nargs="+",
        default=["openai/gpt-5-mini-2025-08-07", "google/gemini-2.5-flash"],
        help="Validation model IDs (held out from calibration)",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("data/questions/questions_all.json"),
        help="Questions file with difficulty scores",
    )
    parser.add_argument(
        "--world-reports-dir",
        type=Path,
        default=Path("data/questions"),
        help="Directory containing game world reports",
    )
    parser.add_argument(
        "--questions-per-tertile",
        type=int,
        default=30,
        help="Number of questions to sample per difficulty tertile",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180,
        help="Timeout per model query in seconds",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for validation results (JSON)",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("data/evaluations/plots"),
        help="Directory for output plots",
    )
    parser.add_argument(
        "--gcp-project",
        help="GCP project ID for Secret Manager (overrides GCP_PROJECT_ID env var)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CivBench Difficulty Calibration Validation")
    print("=" * 70)

    # Configure API keys
    if args.gcp_project:
        os.environ["GCP_PROJECT_ID"] = args.gcp_project
    logger.info("Configuring API keys...")
    configure_api_keys(from_gcp=True)

    # Load questions with difficulty
    logger.info(f"Loading questions from {args.questions_file}...")
    questions = load_questions_with_difficulty(args.questions_file)
    logger.info(f"Found {len(questions)} questions with difficulty scores")

    # Compute tertile boundaries
    logger.info("\nComputing difficulty tertiles...")
    tertile_bounds = compute_tertiles(questions)
    for name, (low, high) in tertile_bounds.items():
        logger.info(f"  {name.capitalize()}: {low:.4f} - {high:.4f}")

    # Sample by tertile
    logger.info(f"\nSampling {args.questions_per_tertile} questions per tertile (seed={args.seed})...")
    sampled = sample_by_tertile(questions, tertile_bounds, args.questions_per_tertile, args.seed)

    # Flatten sampled questions
    all_sampled = []
    for tertile_questions in sampled.values():
        all_sampled.extend(tertile_questions)

    logger.info(f"\nTotal sampled questions: {len(all_sampled)}")

    # Create validation models
    logger.info(f"\nCreating validation models: {args.validation_models}")
    models = get_models(args.validation_models)

    # Create rate limiter
    rate_limiter = ProviderRateLimiter()

    # Run validation
    logger.info(f"\nRunning validation evaluation (timeout={args.timeout}s)...")
    results = asyncio.run(run_validation(
        questions=all_sampled,
        models=models,
        rate_limiter=rate_limiter,
        world_reports_dir=args.world_reports_dir,
        timeout=args.timeout,
    ))

    # Compute metrics
    logger.info("\nComputing validation metrics...")
    metrics = compute_validation_metrics(results, tertile_bounds)

    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    plots = generate_visualizations(metrics, args.plots_dir)

    # Print results
    print("\n" + "=" * 70)
    print("Validation Results")
    print("=" * 70)

    print(f"\nQuestions evaluated: {metrics.get('n_questions', 0)}")

    print(f"\nPearson Correlation (difficulty vs Brier):")
    print(f"  r = {metrics.get('pearson_r', 'N/A'):.4f}")
    print(f"  p-value = {metrics.get('pearson_p', 'N/A'):.4f}")

    print(f"\nPer-Tertile Brier Scores:")
    for tertile in ["easy", "medium", "hard"]:
        tertile_stats = metrics.get("per_tertile", {}).get(tertile, {})
        mean = tertile_stats.get("mean")
        std = tertile_stats.get("std")
        median = tertile_stats.get("median")
        n = tertile_stats.get("n", 0)
        if mean is not None:
            print(f"  {tertile.capitalize():8s}: mean={mean:.4f}, std={std:.4f}, median={median:.4f} (n={n})")
        else:
            print(f"  {tertile.capitalize():8s}: N/A")

    if plots:
        print(f"\nPlots saved to: {args.plots_dir}")
        for name, path in plots.items():
            print(f"  - {name}: {path.name}")

    print("=" * 70)

    # Save results
    if args.output:
        def convert_numpy(obj):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(v) for v in obj]
            return obj

        # Prepare per-tertile without raw brier_scores (keep it cleaner)
        per_tertile_summary = {}
        for t, t_stats in metrics.get("per_tertile", {}).items():
            per_tertile_summary[t] = {
                k: v for k, v in t_stats.items() if k != "brier_scores"
            }

        output_data = convert_numpy({
            "timestamp": datetime.now().isoformat(),
            "validation_models": args.validation_models,
            "questions_per_tertile": args.questions_per_tertile,
            "seed": args.seed,
            "tertile_bounds": {k: list(v) for k, v in tertile_bounds.items()},
            "pearson_r": metrics.get("pearson_r"),
            "pearson_p": metrics.get("pearson_p"),
            "n_questions": metrics.get("n_questions"),
            "per_tertile": per_tertile_summary,
            "questions": metrics.get("questions", []),  # Question-level data
            "plots": {name: str(path) for name, path in plots.items()},
        })
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
