#!/usr/bin/env python3
"""Run H0 continuous model evaluations for the aggregate human comparison.

The existing model run used for the human aggregate proxy has 660 primary
questions: 11 worlds x 5 civilizations x 3 continuous target families x 4
future horizons. H0 is a zero-horizon comprehension analogue, so the H0 set is
the same world/civilization/target combinations at the snapshot turn: 165
questions.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
load_dotenv(REPO_ROOT / ".env")

from civrealm.evaluation.models import get_models, load_api_keys_from_gcp
from civrealm.evaluation.parallel_evaluator import evaluate_continuous_question_batch
from civrealm.evaluation.rate_limiter import ProviderRateLimiter
from civrealm.metrics import compute_crps, compute_mae


PRIMARY_TEMPLATES = {
    "cities_count_continuous": {
        "metric": "cities_count",
        "human_template": "cities",
        "value_range": 40.0,
        "question": "How many cities does {civ} have at turn {turn}?",
    },
    "techs_continuous": {
        "metric": "techs_known",
        "human_template": "techs",
        "value_range": 60.0,
        "question": "How many technologies has {civ} discovered by turn {turn}?",
    },
    "treasury_continuous": {
        "metric": "treasury",
        "human_template": "treasury",
        "value_range": 2000.0,
        "question": "How much gold does {civ} have at turn {turn}?",
    },
}

PRIMARY_HORIZONS = {"H1", "H3", "H4", "H6"}
TURN_TO_HORIZON = {90: "H1", 120: "H2", 150: "H3", 180: "H4", 210: "H5", 240: "H6", 270: "H7"}
DEFAULT_EXCLUDED_MODELS = {
    "anthropic/claude-3-haiku-20240307",
    "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/gemini-2.0-flash-lite-001",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-run",
        type=Path,
        default=Path("data/evaluations/runs/final_2026-03-07_newq_uncond_nonh0_bin_plus_cont_all30.json"),
        help="Existing non-H0 model run used to identify models and primary question combinations.",
    )
    parser.add_argument(
        "--games-dir",
        type=Path,
        default=Path("data/games"),
        help="Directory containing seed*_data.json files with turn-60 ground truth.",
    )
    parser.add_argument(
        "--world-report-dir",
        type=Path,
        default=Path("data/questions"),
        help="Directory containing {game_id}/world_report/turn_060_report.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/human_baseline/aggregate_model_comparison/h0_continuous_eval"),
        help="Directory for H0 questions, model outputs, and scores.",
    )
    parser.add_argument("--snapshot-turn", type=int, default=60)
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Model IDs to run. Defaults to models from --model-run, excluding known unavailable "
            "legacy models. Explicit --models values are not filtered."
        ),
    )
    parser.add_argument("--only-missing", action="store_true", help="Skip model/world batches already present in results.json.")
    parser.add_argument("--dry-run", action="store_true", help="Write question manifest and prompt previews without API calls.")
    parser.add_argument("--write-prompts", action="store_true", help="Write one prompt preview per world.")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per model call in seconds.")
    parser.add_argument("--concurrent-worlds", type=int, default=1)
    parser.add_argument("--load-gcp-secrets", action="store_true")
    parser.add_argument(
        "--gcp-project-id",
        default=None,
        help="GCP project for Secret Manager. Defaults to GOOGLE_CLOUD_PROJECT or GCP_PROJECT_ID.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def setup_logger(output_dir: Path, verbose: bool) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("h0_continuous_eval")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(output_dir / "h0_run.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        seen = set()
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_horizon(question_text: str) -> str | None:
    match = re.search(r"turn\s+(\d+)", question_text)
    if not match:
        return None
    return TURN_TO_HORIZON.get(int(match.group(1)))


def civ_from_question(template_id: str, question_text: str) -> str:
    patterns = {
        "cities_count_continuous": r"^How many cities will (.+) have at turn \d+\?$",
        "techs_continuous": r"^How many technologies will (.+) have discovered by turn \d+\?$",
        "treasury_continuous": r"^How much gold will (.+) have at turn \d+\?$",
    }
    match = re.match(patterns[template_id], question_text)
    if not match:
        raise ValueError(f"Unexpected question text for {template_id}: {question_text}")
    return match.group(1)


def primary_combinations(model_run: dict) -> list[dict]:
    combos = {}
    for question in model_run["questions"]:
        template_id = question.get("template_id")
        if question.get("question_type") != "continuous" or template_id not in PRIMARY_TEMPLATES:
            continue
        horizon = infer_horizon(question["question_text"])
        if horizon not in PRIMARY_HORIZONS:
            continue
        civ = civ_from_question(template_id, question["question_text"])
        key = (question["game_id"], civ, template_id)
        combos[key] = {"game_id": question["game_id"], "civilization": civ, "template_id": template_id}
    return [combos[key] for key in sorted(combos)]


def player_id_for_civ(game_data: dict, civ_name: str) -> str:
    for player_id, info in game_data["civilizations"].items():
        if info["name"] == civ_name:
            return str(player_id)
    raise ValueError(f"Civilization {civ_name!r} not found in game data")


def build_h0_questions(model_run: dict, games_dir: Path, snapshot_turn: int) -> list[dict]:
    questions = []
    for idx, combo in enumerate(primary_combinations(model_run)):
        game_id = combo["game_id"]
        template_id = combo["template_id"]
        spec = PRIMARY_TEMPLATES[template_id]
        game_data = read_json(games_dir / f"{game_id}_data.json")
        player_id = player_id_for_civ(game_data, combo["civilization"])
        ground_truth = game_data["time_series"][spec["metric"]][str(snapshot_turn)][player_id]
        questions.append(
            {
                "question_id": f"h0c{idx:04d}",
                "source_key": f"{game_id}:{combo['civilization']}:{template_id}",
                "game_id": game_id,
                "template_id": f"h0_{template_id}",
                "source_template_id": template_id,
                "human_template": spec["human_template"],
                "question_type": "continuous",
                "horizon": "H0",
                "resolution_turn": snapshot_turn,
                "civilization": combo["civilization"],
                "player_id": player_id,
                "question_text": spec["question"].format(civ=combo["civilization"], turn=snapshot_turn),
                "ground_truth": float(ground_truth),
                "value_range": spec["value_range"],
            }
        )
    return questions


def load_world_report(world_report_dir: Path, game_id: str, snapshot_turn: int) -> str:
    turn = f"{snapshot_turn:03d}"
    candidates = [
        world_report_dir / game_id / "world_report" / f"turn_{turn}_report.txt",
        world_report_dir / game_id / "world_report" / f"turn_{snapshot_turn}_report.txt",
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"No world report found for {game_id}: {candidates}")


def group_by_world(questions: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for question in questions:
        grouped[question["game_id"]].append(question)
    return dict(sorted(grouped.items()))


def load_existing_results(path: Path) -> dict[str, dict] | None:
    if not path.exists():
        return None
    existing = read_json(path)
    return {question["question_id"]: question for question in existing.get("questions", [])}


def merge_results(
    questions: list[dict],
    new_results: list[dict],
    existing_by_question: dict[str, dict] | None,
) -> list[dict]:
    merged = {}
    if existing_by_question:
        merged.update(existing_by_question)
    for result in new_results:
        existing = merged.get(result["question_id"])
        if existing:
            predictions = existing.setdefault("predictions", {})
            predictions.update(result.get("predictions", {}))
            existing.update({k: v for k, v in result.items() if k != "predictions"})
        else:
            merged[result["question_id"]] = result
    return [merged[q["question_id"]] for q in questions if q["question_id"] in merged]


def prediction_complete(result: dict | None, model_id: str) -> bool:
    if not result:
        return False
    percentiles = result.get("predictions", {}).get(model_id, {}).get("percentiles")
    return isinstance(percentiles, dict) and all(key in percentiles for key in ["p10", "p25", "p50", "p75", "p90"])


def model_ids_needed_for_batch(batch: list[dict], model_ids: list[str], existing_by_question: dict[str, dict] | None) -> list[str]:
    if not existing_by_question:
        return model_ids
    needed = []
    for model_id in model_ids:
        if any(not prediction_complete(existing_by_question.get(q["question_id"]), model_id) for q in batch):
            needed.append(model_id)
    return needed


def add_question_metadata(results: list[dict], batch: list[dict]) -> None:
    metadata = {question["question_id"]: question for question in batch}
    for result in results:
        source = metadata[result["question_id"]]
        result.update(
            {
                "source_key": source["source_key"],
                "source_template_id": source["source_template_id"],
                "human_template": source["human_template"],
                "horizon": source["horizon"],
                "resolution_turn": source["resolution_turn"],
                "civilization": source["civilization"],
                "player_id": source["player_id"],
                "value_range": source["value_range"],
            }
        )


async def evaluate_batches(
    questions: list[dict],
    model_ids: list[str],
    world_reports: dict[str, str],
    output_dir: Path,
    only_missing: bool,
    timeout: int,
    concurrent_worlds: int,
    verbose: bool,
    logger: logging.Logger,
) -> list[dict]:
    result_path = output_dir / "h0_results.json"
    existing_by_question = load_existing_results(result_path) if only_missing else None
    rate_limiter = ProviderRateLimiter()
    semaphore = asyncio.Semaphore(concurrent_worlds)
    all_new_results = []

    async def guarded(world_id: str, batch: list[dict]) -> list[dict]:
        batch_model_ids = model_ids_needed_for_batch(batch, model_ids, existing_by_question)
        if not batch_model_ids:
            logger.info("World %s already complete; skipping", world_id)
            return []
        async with semaphore:
            logger.info("World %s: evaluating %d questions with %d models", world_id, len(batch), len(batch_model_ids))
            models = get_models(batch_model_ids)
            results = await evaluate_continuous_question_batch(
                questions=batch,
                models=models,
                rate_limiter=rate_limiter,
                world_report=world_reports[world_id],
                timeout=timeout,
                verbose=verbose,
            )
            add_question_metadata(results, batch)
            return results

    grouped = group_by_world(questions)
    nested = await asyncio.gather(*(guarded(world_id, batch) for world_id, batch in grouped.items()))
    for batch_results in nested:
        all_new_results.extend(batch_results)
    return merge_results(questions, all_new_results, existing_by_question)


def flatten_predictions(results: list[dict], model_ids: list[str]) -> list[dict]:
    rows = []
    for result in results:
        for model_id in model_ids:
            pred = result.get("predictions", {}).get(model_id, {})
            percentiles = pred.get("percentiles") or {}
            rows.append(
                {
                    "model": model_id,
                    "question_id": result["question_id"],
                    "game_id": result["game_id"],
                    "template_id": result["template_id"],
                    "source_template_id": result["source_template_id"],
                    "human_template": result["human_template"],
                    "civilization": result["civilization"],
                    "ground_truth": result["ground_truth"],
                    "p10": percentiles.get("p10"),
                    "p25": percentiles.get("p25"),
                    "p50": percentiles.get("p50"),
                    "p75": percentiles.get("p75"),
                    "p90": percentiles.get("p90"),
                    "latency_ms": pred.get("latency_ms"),
                    "error": pred.get("error"),
                }
            )
    return rows


def score_results(results: list[dict], model_ids: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    score_rows = []
    for result in results:
        for model_id in model_ids:
            pred = result.get("predictions", {}).get(model_id, {})
            percentiles = pred.get("percentiles")
            if not isinstance(percentiles, dict) or not all(key in percentiles for key in ["p10", "p25", "p50", "p75", "p90"]):
                continue
            crps = compute_crps(percentiles, result["ground_truth"])
            mae = compute_mae(percentiles["p50"], result["ground_truth"])
            score_rows.append(
                {
                    "model": model_id,
                    "question_id": result["question_id"],
                    "game_id": result["game_id"],
                    "template_id": result["template_id"],
                    "source_template_id": result["source_template_id"],
                    "human_template": result["human_template"],
                    "civilization": result["civilization"],
                    "ground_truth": result["ground_truth"],
                    "crps": crps,
                    "mae": mae,
                    "normalized_crps": crps / result["value_range"],
                    "normalized_mae": mae / result["value_range"],
                }
            )

    summary_rows = []
    by_model = defaultdict(list)
    for row in score_rows:
        by_model[row["model"]].append(row)
    for model_id in model_ids:
        rows = by_model.get(model_id, [])
        if not rows:
            summary_rows.append(
                {
                    "model": model_id,
                    "group": "overall",
                    "value": "all",
                    "n_questions": 165,
                    "n_valid_predictions": 0,
                    "crps": "",
                    "mae": "",
                    "normalized_crps": "",
                    "normalized_mae": "",
                }
            )
            continue
        summary_rows.append(summarize_rows(model_id, rows, "overall", "all"))
        for key in ["human_template", "game_id"]:
            for value in sorted({row[key] for row in rows}):
                summary_rows.append(summarize_rows(model_id, [row for row in rows if row[key] == value], key, value))

    template_rows = [
        row for row in summary_rows
        if row["group"] in {"overall", "human_template"}
    ]
    return score_rows, summary_rows, template_rows


def summarize_rows(model_id: str, rows: list[dict], group: str, value: str) -> dict:
    return {
        "model": model_id,
        "group": group,
        "value": value,
        "n_questions": len({row["question_id"] for row in rows}),
        "n_valid_predictions": len(rows),
        "crps": mean(row["crps"] for row in rows),
        "mae": mean(row["mae"] for row in rows),
        "normalized_crps": mean(row["normalized_crps"] for row in rows),
        "normalized_mae": mean(row["normalized_mae"] for row in rows),
    }


def write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        """# H0 Continuous Model Evaluation

Generated by `scripts/run_h0_continuous_model_eval.py`.

This is the zero-horizon analogue of the primary aggregate human/model comparison. The source comparison has 660 future questions: 11 worlds x 5 civilizations x 3 target families x 4 horizons. H0 collapses the four future horizons to the snapshot turn, so this run has 165 questions: 11 worlds x 5 civilizations x 3 target families.

Files:

- `h0_questions.csv`: question manifest and turn-60 ground truth.
- `h0_results.json`: raw model predictions by question.
- `h0_question_predictions.csv`: flattened p10/p25/p50/p75/p90 outputs.
- `h0_forecast_scores.csv`: CRPS and MAE per model-question prediction.
- `h0_model_summary.csv`: overall, target-template, and world summaries.
- `h0_template_summary.csv`: compact overall and target-template summaries.
- `h0_metadata.json`: model list, source run, and run metadata.

The H0 questions are comprehension questions: the answer is available in the turn-60 world report. They should be interpreted as a report-reading sanity check, not a forecasting comparison.
""",
        encoding="utf-8",
    )


def write_prompt_previews(output_dir: Path, questions: list[dict], world_reports: dict[str, str]) -> None:
    from civrealm.evaluation.parallel_evaluator import build_continuous_batch_prompt

    prompt_dir = output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for world_id, batch in group_by_world(questions).items():
        (prompt_dir / f"{world_id}_h0_prompt.txt").write_text(
            build_continuous_batch_prompt(batch, world_reports[world_id]),
            encoding="utf-8",
        )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir, args.verbose)

    model_run = read_json(args.model_run)
    if args.models:
        model_ids = args.models
        excluded_default_models = []
    else:
        excluded_default_models = [
            model_id
            for model_id in model_run["metadata"]["models"]
            if model_id in DEFAULT_EXCLUDED_MODELS
        ]
        model_ids = [
            model_id
            for model_id in model_run["metadata"]["models"]
            if model_id not in DEFAULT_EXCLUDED_MODELS
        ]
    questions = build_h0_questions(model_run, args.games_dir, args.snapshot_turn)
    if len(questions) != 165:
        raise ValueError(f"Expected 165 H0 questions, got {len(questions)}")

    world_reports = {
        world_id: load_world_report(args.world_report_dir, world_id, args.snapshot_turn)
        for world_id in sorted({question["game_id"] for question in questions})
    }

    question_rows = [
        {
            key: question[key]
            for key in [
                "question_id",
                "source_key",
                "game_id",
                "template_id",
                "source_template_id",
                "human_template",
                "horizon",
                "resolution_turn",
                "civilization",
                "player_id",
                "question_text",
                "ground_truth",
                "value_range",
            ]
        }
        for question in questions
    ]
    write_csv(args.output_dir / "h0_questions.csv", question_rows)
    if args.write_prompts or args.dry_run:
        write_prompt_previews(args.output_dir, questions, world_reports)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_model_run": str(args.model_run),
        "source_model_run_id": model_run.get("metadata", {}).get("run_id"),
        "snapshot_turn": args.snapshot_turn,
        "n_questions": len(questions),
        "n_worlds": len(world_reports),
        "models": model_ids,
        "n_models": len(model_ids),
        "excluded_default_models": excluded_default_models,
        "interpretation": "H0 comprehension analogue of the 660-question primary aggregate comparison; not a future forecast.",
    }
    (args.output_dir / "h0_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_readme(args.output_dir)

    logger.info("Prepared %d H0 continuous questions across %d worlds for %d models", len(questions), len(world_reports), len(model_ids))
    if args.dry_run:
        logger.info("Dry run complete; no model APIs called")
        return 0

    if args.load_gcp_secrets:
        project_id = args.gcp_project_id or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID")
        load_api_keys_from_gcp(project_id=project_id)

    results = asyncio.run(
        evaluate_batches(
            questions=questions,
            model_ids=model_ids,
            world_reports=world_reports,
            output_dir=args.output_dir,
            only_missing=args.only_missing,
            timeout=args.timeout,
            concurrent_worlds=args.concurrent_worlds,
            verbose=args.verbose,
            logger=logger,
        )
    )

    output = {"metadata": metadata, "questions": results}
    (args.output_dir / "h0_results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "h0_question_predictions.csv", flatten_predictions(results, model_ids))
    score_rows, summary_rows, template_rows = score_results(results, model_ids)
    write_csv(args.output_dir / "h0_forecast_scores.csv", score_rows)
    write_csv(args.output_dir / "h0_model_summary.csv", summary_rows)
    write_csv(args.output_dir / "h0_template_summary.csv", template_rows)
    logger.info("Wrote H0 results to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
