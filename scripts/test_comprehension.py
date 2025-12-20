#!/usr/bin/env python3
"""
Run comprehension checks on world reports without forecasting.

Example:
    python scripts/test_comprehension.py --data-dir data/questions --models gpt-4.1-2025-04-14
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.llm.model_registry import configure_api_keys, MODELS
from civrealm.evaluation.rate_limiter import ProviderRateLimiter
from civrealm.evaluation.comprehension import (
    run_comprehension_checks,
    find_world_report,
)

# Default model sets (same as main evaluator)
FORECASTBENCH_MODELS = [
    "o3-2025-04-16",
    "gpt-4.1-2025-04-14",
    "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

FRONTIER_MODELS = [
    "gemini-3-pro-preview",
    "gpt-5.1-2025-11-13",
]


def setup_logging(run_id: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("civbench_comp")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_dir / "main.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(file_handler)

    return logger


def discover_games(data_dir: Path) -> list[str]:
    """Discover games under data_dir, or treat data_dir itself as the game."""
    # If data_dir itself looks like a game folder, use it directly
    if find_world_report(data_dir) or any(data_dir.glob("turn_*_data.json")):
        return [data_dir.name]

    # Otherwise, scan subdirectories (original behavior)
    game_ids = []
    for p in data_dir.iterdir():
        if not p.is_dir():
            continue
        if find_world_report(p) or any(p.glob("turn_*_data.json")):
            game_ids.append(p.name)
    return sorted(game_ids)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run comprehension checks on world reports.")
    parser.add_argument("--data-dir", type=str, default="data/questions", help="Directory containing game folders with world_report/")
    parser.add_argument("--models", nargs="+", default=None, help="Model IDs to use (default: ForecastBench + frontier)")
    parser.add_argument("--forecastbench-only", action="store_true", help="Use only ForecastBench models")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per model query in seconds (0 = no timeout)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging of prompts/responses")
    parser.add_argument("--output", type=str, help="Optional output JSON path for results")
    parser.add_argument("--dry-run", action="store_true", help="List targets but do not query models")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / "comp" / f"comp_{run_id}"
    logger = setup_logging(run_id, log_dir)

    load_dotenv()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data dir not found: {data_dir}")
        return 1

    game_ids = discover_games(data_dir)
    logger.info(f"Found {len(game_ids)} games with world reports in {data_dir}")
    if not game_ids:
        return 1

    if args.dry_run:
        for gid in game_ids:
            logger.info(f"  - {gid}")
        return 0

    # Configure API keys
    logger.info("Configuring API keys...")
    configure_api_keys(from_gcp=True)

    # Select models
    if args.models:
        model_ids = args.models
    elif args.forecastbench_only:
        model_ids = FORECASTBENCH_MODELS
    else:
        model_ids = FORECASTBENCH_MODELS + FRONTIER_MODELS

    models = []
    logger.info(f"Finding models ({len(model_ids)} requested)...")
    for mid in model_ids:
        m = next((m for m in MODELS if mid in m.id or m.id in mid), None)
        if m:
            models.append(m)
            logger.info(f"  Found: {mid} -> {m.id}")
        else:
            logger.warning(f"  Not found: {mid}")

    if not models:
        logger.error("No valid models found")
        return 1

    timeout = args.timeout if args.timeout > 0 else None
    rate_limiter = ProviderRateLimiter()

    logger.info(f"Running comprehension checks on {len(game_ids)} games with {len(models)} models...")
    comp_results, comp_summary = await run_comprehension_checks(
        game_ids=game_ids,
        models=models,
        rate_limiter=rate_limiter,
        data_dir=data_dir,
        timeout=timeout,
        verbose=args.verbose,
        logger=logger,
        batch_size=11,
    )

    logger.info("\nComprehension accuracy:")
    for mid, stats in comp_summary.items():
        total = stats.get("total", 0)
        correct = stats.get("correct", 0)
        acc = stats.get("accuracy", float("nan"))
        if total:
            logger.info(f"  {mid}: {correct}/{total} ({acc:.2%})")
        else:
            logger.info(f"  {mid}: no answers")

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("logs") / "comprehension" / f"comprehension_{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Reorder questions: incorrect first
    reordered = []
    correct = []
    incorrect_counts: dict[str, int] = {}
    for q in comp_results:
        # If any model probability is None, treat as incorrect
        any_incorrect = False
        for pred in q.get("predictions", {}).values():
            prob = pred.get("probability")
            truth = bool(q.get("ground_truth"))
            if prob is None or (prob >= 0.5) != truth:
                any_incorrect = True
                break
        if any_incorrect:
            reordered.append(q)
            tid = q.get("template_id", "unknown")
            incorrect_counts[tid] = incorrect_counts.get(tid, 0) + 1
        else:
            correct.append(q)
    reordered.extend(correct)

    payload = {
        "metadata": {
            "run_id": run_id,
            "data_dir": str(data_dir),
            "models": [m.id for m in models],
            "games": game_ids,
            "start_time": run_id,
            "output_path": str(out_path),
            "incorrect_by_template": incorrect_counts,
        },
        "comprehension_summary": comp_summary,
        "questions": reordered,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
