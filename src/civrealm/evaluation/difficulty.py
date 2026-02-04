"""
Empirical difficulty computation from evaluation results.

This module computes difficulty scores for questions based on actual model performance.
Difficulty is computed as the mean Brier score across all model evaluations,
with equal weighting per model (not per evaluation run).
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..world_reports.questions.schema import EmpiricalDifficulty
from ..world_reports.questions.io import (
    load_question_bank,
    save_question_bank,
    _dict_to_empirical_difficulty,
    _empirical_difficulty_to_dict,
)


DIFFICULTY_VERSION = "v1.0"


def compute_brier_score(probability: float, ground_truth: bool) -> float:
    """
    Compute Brier score for a single prediction.

    Args:
        probability: Predicted probability (0.0-1.0)
        ground_truth: Actual outcome (True/False)

    Returns:
        Brier score (0.0-1.0, lower is better)
    """
    return (probability - int(ground_truth)) ** 2


def aggregate_evaluations(eval_dir: Path) -> dict[str, dict[str, list[float]]]:
    """
    Aggregate all evaluation results by question_id and model.

    Loads all evaluation JSON files from the directory and merges results
    by question_id, grouping Brier scores by model. Handles the same question
    appearing in multiple evaluation runs.

    Args:
        eval_dir: Directory containing evaluation JSON files

    Returns:
        Nested dict: {question_id: {model_id: [brier_scores...]}}
        Also includes 'ground_truth' key per question for validation.
    """
    # Structure: {question_id: {model_id: [brier_scores...], '_ground_truth': bool, '_game_id': str}}
    aggregated: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(list))

    eval_files = list(eval_dir.glob("*.json"))
    print(f"Found {len(eval_files)} evaluation files")

    for eval_file in eval_files:
        try:
            with open(eval_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {eval_file}: {e}")
            continue

        questions = data.get("questions", [])

        for q in questions:
            question_id = q.get("question_id")
            game_id = q.get("game_id")
            ground_truth = q.get("ground_truth")

            if question_id is None or ground_truth is None:
                continue

            # Store metadata (game_id, ground_truth) for later use
            # Use composite key to handle same question_id across different games
            composite_key = f"{game_id}:{question_id}"
            aggregated[composite_key]["_ground_truth"] = ground_truth
            aggregated[composite_key]["_game_id"] = game_id
            aggregated[composite_key]["_question_id"] = question_id

            predictions = q.get("predictions", {})
            for model_id, pred_data in predictions.items():
                probability = pred_data.get("probability")
                error = pred_data.get("error")

                # Skip failed predictions
                if probability is None or error is not None:
                    continue

                brier = compute_brier_score(probability, ground_truth)
                aggregated[composite_key][model_id].append(brier)

    return dict(aggregated)


def compute_question_difficulties(
    aggregated: dict[str, dict[str, Any]],
    min_evaluations: int = 1,
) -> dict[str, EmpiricalDifficulty]:
    """
    Compute empirical difficulty for all questions.

    Algorithm:
    1. For each question, compute per-model mean Brier score
    2. Difficulty score = mean of per-model means (equal model weighting)
    3. Track num_evaluations, num_models, model_scores for transparency

    Args:
        aggregated: Output from aggregate_evaluations()
        min_evaluations: Minimum number of total evaluations required

    Returns:
        Dict mapping composite_key to EmpiricalDifficulty
    """
    difficulties: dict[str, EmpiricalDifficulty] = {}
    timestamp = datetime.now(timezone.utc).isoformat()

    for composite_key, data in aggregated.items():
        # Separate metadata from model scores
        model_scores: dict[str, float] = {}
        total_evaluations = 0

        for key, value in data.items():
            if key.startswith("_"):
                continue  # Skip metadata keys

            # value is list of Brier scores for this model
            if isinstance(value, list) and len(value) > 0:
                model_scores[key] = sum(value) / len(value)
                total_evaluations += len(value)

        # Skip if insufficient evaluations
        if total_evaluations < min_evaluations or len(model_scores) == 0:
            continue

        # Compute overall difficulty as mean of per-model means
        difficulty_score = sum(model_scores.values()) / len(model_scores)

        difficulties[composite_key] = EmpiricalDifficulty(
            score=difficulty_score,
            percentile=None,  # Computed separately after all scores are known
            num_evaluations=total_evaluations,
            num_models=len(model_scores),
            model_scores=model_scores,
            last_updated=timestamp,
            version=DIFFICULTY_VERSION,
        )

    return difficulties


def compute_percentile_ranks(
    difficulties: dict[str, EmpiricalDifficulty]
) -> dict[str, EmpiricalDifficulty]:
    """
    Add percentile ranks to difficulty scores.

    Percentile indicates what fraction of questions are easier (lower score).
    100th percentile = hardest question.

    Args:
        difficulties: Dict from compute_question_difficulties()

    Returns:
        Same dict with percentile field populated
    """
    if not difficulties:
        return difficulties

    # Get valid scores and sort
    scores_with_keys = [
        (key, d.score)
        for key, d in difficulties.items()
        if d.score is not None
    ]

    if not scores_with_keys:
        return difficulties

    # Sort by score (ascending - lower score = easier)
    sorted_items = sorted(scores_with_keys, key=lambda x: x[1])
    n = len(sorted_items)

    # Compute percentile for each
    for rank, (key, _) in enumerate(sorted_items):
        # Percentile: what fraction of questions have lower (easier) scores
        percentile = (rank / n) * 100
        difficulties[key].percentile = percentile

    return difficulties


def update_question_files(
    questions_dir: Path,
    difficulties: dict[str, EmpiricalDifficulty],
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Update question JSON files with computed difficulty scores.

    Handles two formats:
    1. questions_all.json at the root (game_id in parameters.game_id)
    2. Per-game directories with questions.json (legacy format)

    Args:
        questions_dir: Directory containing question files
        difficulties: Dict mapping "game_id:question_id" to EmpiricalDifficulty
        dry_run: If True, don't actually save files

    Returns:
        Tuple of (questions_updated, files_updated)
    """
    questions_updated = 0
    files_updated = 0

    # First, try questions_all.json at the root
    questions_all_file = questions_dir / "questions_all.json"
    if questions_all_file.exists():
        with open(questions_all_file) as f:
            data = json.load(f)

        file_modified = False
        for q in data.get("questions", []):
            question_id = q.get("question_id")
            # game_id is in parameters.game_id
            game_id = q.get("parameters", {}).get("game_id")
            if game_id and question_id:
                composite_key = f"{game_id}:{question_id}"
                if composite_key in difficulties:
                    diff = difficulties[composite_key]
                    q["empirical_difficulty"] = {
                        "score": diff.score,
                        "percentile": diff.percentile,
                        "num_evaluations": diff.num_evaluations,
                        "num_models": diff.num_models,
                        "model_scores": diff.model_scores,
                        "last_updated": diff.last_updated,
                        "version": diff.version,
                    }
                    questions_updated += 1
                    file_modified = True

        if file_modified and not dry_run:
            with open(questions_all_file, "w") as f:
                json.dump(data, f, indent=2)
            files_updated += 1

        return questions_updated, files_updated

    # Fallback: Process per-game directories (legacy format)
    by_game: dict[str, dict[str, EmpiricalDifficulty]] = defaultdict(dict)
    for composite_key, diff in difficulties.items():
        game_id, question_id = composite_key.split(":", 1)
        by_game[game_id][question_id] = diff

    for game_dir in sorted(questions_dir.iterdir()):
        if not game_dir.is_dir():
            continue

        game_id = game_dir.name
        if game_id not in by_game:
            continue

        game_difficulties = by_game[game_id]
        file_modified = False

        for qfile_name in ["questions.json", "h0_questions.json"]:
            questions_file = game_dir / qfile_name
            if not questions_file.exists():
                continue

            with open(questions_file) as f:
                data = json.load(f)

            for q in data.get("questions", []):
                question_id = q.get("question_id")
                if question_id in game_difficulties:
                    diff = game_difficulties[question_id]
                    q["empirical_difficulty"] = {
                        "score": diff.score,
                        "percentile": diff.percentile,
                        "num_evaluations": diff.num_evaluations,
                        "num_models": diff.num_models,
                        "model_scores": diff.model_scores,
                        "last_updated": diff.last_updated,
                        "version": diff.version,
                    }
                    questions_updated += 1
                    file_modified = True

            if file_modified and not dry_run:
                with open(questions_file, "w") as f:
                    json.dump(data, f, indent=2)

        if file_modified:
            files_updated += 1

    return questions_updated, files_updated


def get_difficulty_statistics(
    difficulties: dict[str, EmpiricalDifficulty]
) -> dict[str, Any]:
    """
    Compute summary statistics for difficulty scores.

    Args:
        difficulties: Dict from compute_question_difficulties()

    Returns:
        Dict with statistics including min, max, mean, median, quartiles
    """
    scores = [d.score for d in difficulties.values() if d.score is not None]

    if not scores:
        return {"count": 0}

    scores_sorted = sorted(scores)
    n = len(scores_sorted)

    def percentile(p: float) -> float:
        k = (n - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < n else f
        return scores_sorted[f] + (k - f) * (scores_sorted[c] - scores_sorted[f])

    # Count by model
    model_counts: dict[str, int] = defaultdict(int)
    for d in difficulties.values():
        for model in d.model_scores.keys():
            model_counts[model] += 1

    return {
        "count": n,
        "min": min(scores),
        "max": max(scores),
        "mean": sum(scores) / n,
        "median": percentile(50),
        "p25": percentile(25),
        "p75": percentile(75),
        "p10": percentile(10),
        "p90": percentile(90),
        "models_contributing": dict(model_counts),
    }


def get_tertile_bounds(scores: list[float]) -> dict[str, tuple[float, float]]:
    """
    Compute difficulty tertile boundaries.

    Args:
        scores: List of difficulty scores

    Returns:
        Dict with 'easy', 'medium', 'hard' keys, each mapping to (min, max) range
    """
    if not scores:
        return {"easy": (0.0, 0.33), "medium": (0.33, 0.67), "hard": (0.67, 1.0)}

    scores_sorted = sorted(scores)
    n = len(scores_sorted)

    def percentile(p: float) -> float:
        k = (n - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < n else f
        return scores_sorted[f] + (k - f) * (scores_sorted[c] - scores_sorted[f])

    p33 = percentile(33.33)
    p67 = percentile(66.67)

    return {
        "easy": (0.0, p33),
        "medium": (p33, p67),
        "hard": (p67, 1.0),
    }


def filter_by_tertile(
    difficulties: dict[str, EmpiricalDifficulty],
    tertile: str,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, EmpiricalDifficulty]:
    """
    Filter difficulties to only include questions in a specific tertile.

    Args:
        difficulties: Dict from compute_question_difficulties()
        tertile: One of 'easy', 'medium', 'hard'
        bounds: Optional pre-computed tertile bounds

    Returns:
        Filtered dict containing only questions in the specified tertile
    """
    if tertile not in ("easy", "medium", "hard"):
        raise ValueError(f"Invalid tertile: {tertile}. Must be 'easy', 'medium', or 'hard'")

    if bounds is None:
        scores = [d.score for d in difficulties.values() if d.score is not None]
        bounds = get_tertile_bounds(scores)

    low, high = bounds[tertile]

    return {
        key: diff
        for key, diff in difficulties.items()
        if diff.score is not None and low <= diff.score <= high
    }
