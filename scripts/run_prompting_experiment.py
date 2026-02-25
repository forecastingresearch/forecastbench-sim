#!/usr/bin/env python3
"""
Prompting intervention experiment for the Forward Simulation Gap.

Tests whether prompting variants can close the conditional calibration gap.
Runs 5 prompting conditions × 3 models on Republic conditional questions.

Usage:
    uv run python scripts/run_prompting_experiment.py
    uv run python scripts/run_prompting_experiment.py --dry-run
    uv run python scripts/run_prompting_experiment.py --condition cot --models openai/o3-2025-04-16
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from civrealm.evaluation.models import get_models, load_api_keys_from_gcp

load_api_keys_from_gcp()

from civrealm.evaluation.sampling import load_all_questions, stratified_sample_batched
from civrealm.evaluation.rate_limiter import ProviderRateLimiter
from civrealm.evaluation import parallel_evaluator
from civrealm.evaluation.parallel_evaluator import run_batch_evaluation
from civrealm.metrics import compute_brier_score, compute_calibration_error


# Models spanning the conditional gap range
EXPERIMENT_MODELS = [
    "openai/o3-2025-04-16",           # smallest gap
    "anthropic/claude-opus-4-5-20251101",  # mid-range
    "openai/gpt-4.1-2025-04-14",      # largest gap
]

DATA_DIR = Path("data/conditional/republic/conditional")
OUTPUT_DIR = Path("data/results/prompting_experiment")

# Prompting conditions: (name, prefix text inserted before the world report)
PROMPTING_CONDITIONS = {
    "baseline": "",  # no prefix — matches existing conditional eval
    "cot": (
        "IMPORTANT: Think step by step about how the intervention described in "
        "the question affects the game outcome before giving your probability.\n\n"
    ),
    "explicit_updating": (
        "IMPORTANT: The questions below describe a change to the world state. "
        "Given this change, update your beliefs about the following outcomes. "
        "Your predictions should reflect the world AFTER the intervention, not "
        "the default trajectory.\n\n"
    ),
    "structured_reasoning": (
        "IMPORTANT: Before forecasting, follow these steps:\n"
        "1. Describe how the intervention changes the game state\n"
        "2. Identify which outcomes are affected and in which direction\n"
        "3. Then give your probability estimates\n\n"
    ),
    "base_rate_reanchoring": (
        "IMPORTANT: The questions below ask about outcomes under a specific "
        "intervention. First, estimate: under this intervention, what fraction "
        "of similar game scenarios would produce this outcome? Use that base "
        "rate estimate to anchor your probability, then adjust for the specifics "
        "of this game.\n\n"
    ),
    "calibration_nudge": (
        "IMPORTANT: The questions below describe an intervention that changes "
        "the game state. The probability of each outcome may be VERY DIFFERENT "
        "under this intervention than in the default game. Do not assume outcomes "
        "will follow the same pattern as without the intervention. Consider "
        "carefully how the intervention changes the likelihood before forecasting.\n\n"
    ),
}


def make_patched_prompt_builder(prefix: str):
    """Create a patched build_batch_prompt that inserts a prefix."""
    original_builder = parallel_evaluator.build_batch_prompt.__wrapped__ \
        if hasattr(parallel_evaluator.build_batch_prompt, '__wrapped__') \
        else parallel_evaluator.build_batch_prompt

    def patched_build_batch_prompt(questions, world_report):
        # Get the original prompt
        prompt = original_builder(questions, world_report)
        if not prefix:
            return prompt
        # Insert prefix after the first paragraph (system instruction)
        # Find the "## World Report" section and insert before it
        marker = "## World Report"
        if marker in prompt:
            idx = prompt.index(marker)
            return prompt[:idx] + prefix + prompt[idx:]
        else:
            # Fallback: prepend
            return prefix + prompt
    return patched_build_batch_prompt


# Store the original function
_original_build_batch_prompt = parallel_evaluator.build_batch_prompt


def setup_logging():
    logger = logging.getLogger("civbench_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(console)
    return logger


async def run_condition(
    condition_name: str,
    prefix: str,
    question_batches: list,
    model_ids: list[str],
    data_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
):
    """Run one prompting condition across all models."""
    logger = logging.getLogger("civbench_eval")
    logger.info(f"\n{'='*80}")
    logger.info(f"  CONDITION: {condition_name}")
    logger.info(f"  Prefix: {prefix[:80]}..." if prefix else "  Prefix: (none)")
    logger.info(f"{'='*80}")

    if dry_run:
        # Show what the prompt would look like
        patched = make_patched_prompt_builder(prefix)
        sample_prompt = patched(
            [{"question_text": "If X switches to Republic, would X have more treasury than Y at turn 90?"}],
            "[world report would go here]"
        )
        logger.info(f"  DRY RUN — sample prompt:\n{sample_prompt[:500]}...")
        return None

    # Monkeypatch the prompt builder
    parallel_evaluator.build_batch_prompt = make_patched_prompt_builder(prefix)

    try:
        models = get_models(model_ids)
        rate_limiter = ProviderRateLimiter()

        total_qs = sum(len(b) for b in question_batches)
        logger.info(f"  Models: {[m.id for m in models]}")
        logger.info(f"  Questions: {total_qs} across {len(question_batches)} batches")

        results = await run_batch_evaluation(
            question_batches=question_batches,
            models=models,
            rate_limiter=rate_limiter,
            data_dir=data_dir,
            timeout=120,
            concurrent_batches=5,
        )

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"prompting_{condition_name}.json"

        # Compute metrics per model
        model_results = {}
        for model_id in model_ids:
            preds = []
            outcomes = []
            for r in results:
                pred_data = r.get("predictions", {}).get(model_id, {})
                prob = pred_data.get("probability")
                if prob is not None:
                    preds.append(prob)
                    outcomes.append(r["ground_truth"])

            brier = compute_brier_score(preds, outcomes) if preds else float('nan')
            ece_val = compute_calibration_error(preds, outcomes) if preds else float('nan')
            model_results[model_id] = {
                "brier_score": brier,
                "ece": ece_val,
                "num_predictions": len(preds),
            }

        output_data = {
            "metadata": {
                "condition": condition_name,
                "prompt_prefix": prefix,
                "models": model_ids,
                "timestamp": datetime.now().isoformat(),
                "total_questions": total_qs,
            },
            "model_results": model_results,
            "questions": results,
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"  Results saved to {output_file}")
        for mid, mr in model_results.items():
            short = mid.split("/")[-1][:20]
            logger.info(f"    {short}: Brier={mr['brier_score']:.4f}, ECE={mr['ece']:.4f}")

        return output_data

    finally:
        # Restore original function
        parallel_evaluator.build_batch_prompt = _original_build_batch_prompt


async def main():
    parser = argparse.ArgumentParser(description="Prompting intervention experiment")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts without querying")
    parser.add_argument("--condition", type=str, help="Run only this condition")
    parser.add_argument("--models", nargs="+", help="Override model list")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    logger = setup_logging()

    model_ids = args.models or EXPERIMENT_MODELS

    # Load and batch questions
    logger.info("Loading questions...")
    all_questions = load_all_questions(args.data_dir)
    binary_questions = [q for q in all_questions if q.get("question_type") == "binary"]
    logger.info(f"  {len(binary_questions)} binary questions loaded")

    # Batch by game
    batches_by_game = defaultdict(list)
    for q in binary_questions:
        batches_by_game[q["game_id"]].append(q)
    question_batches = list(batches_by_game.values())
    logger.info(f"  {len(question_batches)} game batches")

    # Select conditions
    if args.condition:
        conditions = {args.condition: PROMPTING_CONDITIONS[args.condition]}
    else:
        conditions = PROMPTING_CONDITIONS

    # Run each condition sequentially (models run in parallel within each)
    all_results = {}
    for cond_name, prefix in conditions.items():
        result = await run_condition(
            cond_name, prefix, question_batches, model_ids,
            args.data_dir, args.output_dir, args.dry_run,
        )
        if result:
            all_results[cond_name] = result

    if all_results:
        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info(f"  SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"  {'Condition':<25} {'Model':<25} {'Brier':>8} {'ECE':>8}")
        logger.info(f"  {'-'*70}")
        for cond_name, result in all_results.items():
            for mid, mr in result["model_results"].items():
                short = mid.split("/")[-1][:24]
                logger.info(
                    f"  {cond_name:<25} {short:<25} "
                    f"{mr['brier_score']:>8.4f} {mr['ece']:>8.4f}"
                )


if __name__ == "__main__":
    asyncio.run(main())
