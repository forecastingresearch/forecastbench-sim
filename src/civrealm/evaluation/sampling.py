"""Stratified sampling for CivBench evaluation questions."""

import json
import random
from collections import defaultdict
from pathlib import Path


def load_all_questions(data_dir: Path, template_filter: str | None = None) -> list[dict]:
    """
    Load all questions from game folders.

    Args:
        data_dir: Directory containing game folders with questions.json files
        template_filter: Optional filter to only load questions of a specific template

    Returns:
        List of question dicts with keys: game_id, question_id, question_text,
        ground_truth, parameters, difficulty
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
                "template_id": q.get("template_id"),
                "question_text": q.get("question_text"),
                "ground_truth": bool(answer),
                "parameters": q.get("parameters", {}),
                "difficulty": q.get("difficulty", {}),
            })

    return questions


def stratified_sample(
    questions: list[dict],
    per_difficulty: int,
    seed: int,
    difficulties: list[int] | None = None,
) -> list[dict]:
    """
    Sample questions with balanced representation across difficulty levels.

    Guarantees exactly `per_difficulty` questions from each composite difficulty
    level (2-6), resulting in perfectly balanced samples.

    Args:
        questions: Full list of questions to sample from
        per_difficulty: Number of questions to sample from each difficulty level
        seed: Random seed for reproducibility
        difficulties: Difficulty levels to sample from (default: [2, 3, 4, 5, 6])

    Returns:
        List of sampled questions, balanced across difficulties.
        Total size = per_difficulty * len(difficulties)

    Example:
        >>> questions = load_all_questions(Path("data/questions"))
        >>> sampled = stratified_sample(questions, per_difficulty=20, seed=42)
        >>> len(sampled)  # 20 * 5 = 100 questions
        100
    """
    if difficulties is None:
        difficulties = [2, 3, 4, 5, 6]

    random.seed(seed)
    sampled = []

    # Group questions by composite difficulty
    by_difficulty: dict[int, list[dict]] = defaultdict(list)
    for q in questions:
        d = q.get("difficulty", {}).get("composite", 0)
        by_difficulty[d].append(q)

    # Sample from each difficulty level
    for d in difficulties:
        pool = by_difficulty[d]
        n = min(per_difficulty, len(pool))

        if n < per_difficulty:
            print(f"Warning: Only {len(pool)} questions available for difficulty {d}, "
                  f"requested {per_difficulty}")

        if pool:
            sampled.extend(random.sample(pool, n))

    return sampled


def get_difficulty_distribution(questions: list[dict]) -> dict[int, int]:
    """
    Get the count of questions at each difficulty level.

    Args:
        questions: List of question dicts

    Returns:
        Dict mapping difficulty level to count
    """
    dist: dict[int, int] = defaultdict(int)
    for q in questions:
        d = q.get("difficulty", {}).get("composite", 0)
        dist[d] += 1
    return dict(sorted(dist.items()))


def stratified_sample_batched(
    questions: list[dict],
    per_difficulty: int | None,
    seed: int,
    difficulties: list[int] | None = None,
    min_per_template: int | None = None,
) -> list[list[dict]]:
    """
    Sample questions with balanced difficulty, then group by game for batching.

    Two-phase approach:
    1. Stratified sample: exactly `per_difficulty` questions from each difficulty level
    2. Regroup by game_id: organize sampled questions into batches by game

    This gives both:
    - Guaranteed difficulty balance (exactly N per level)
    - Token efficiency (questions from same game share one world report)

    Args:
        questions: Full list of questions to sample from
        per_difficulty: Number of questions per difficulty level (optional)
        seed: Random seed for reproducibility
        difficulties: Difficulty levels to sample (default: [2, 3, 4, 5, 6])
        min_per_template: Minimum questions to include per template_id (optional)

    Returns:
        List of question batches, grouped by game_id.
        Total questions = per_difficulty * len(difficulties)

    Example:
        >>> batches = stratified_sample_batched(questions, per_difficulty=20, seed=42)
        >>> sum(len(b) for b in batches)  # 20 * 5 = 100 total
        100
        >>> # Each batch contains questions from same game
    """
    # Phase 1: Optional per-template minimums (can be used standalone)
    sampled = []
    remaining_questions = questions
    if min_per_template and min_per_template > 0:
        random.seed(seed)
        by_template: dict[str, list[dict]] = defaultdict(list)
        for q in questions:
            by_template[q.get("template_id", "unknown")].append(q)

        sampled_ids = set()
        for template_id, pool in sorted(by_template.items()):
            n = min(min_per_template, len(pool))
            if n < min_per_template:
                print(
                    f"Warning: Only {len(pool)} questions available for template {template_id}, "
                    f"requested {min_per_template}"
                )
            if n > 0:
                chosen = random.sample(pool, n)
                sampled.extend(chosen)
                for q in chosen:
                    sampled_ids.add((q.get("game_id"), q.get("question_id")))

        remaining_questions = [
            q for q in questions
            if (q.get("game_id"), q.get("question_id")) not in sampled_ids
        ]

        if per_difficulty is None:
            # Return just the per-template minimums, batched by game
            by_game: dict[str, list[dict]] = defaultdict(list)
            for q in sampled:
                by_game[q["game_id"]].append(q)
            return [by_game[gid] for gid in sorted(by_game.keys())]

        if difficulties is None:
            difficulties = [2, 3, 4, 5, 6]

        preselected_by_diff: dict[int, int] = defaultdict(int)
        for q in sampled:
            d = q.get("difficulty", {}).get("composite", 0)
            preselected_by_diff[d] += 1

        remaining_by_diff: dict[int, list[dict]] = defaultdict(list)
        for q in remaining_questions:
            d = q.get("difficulty", {}).get("composite", 0)
            remaining_by_diff[d].append(q)

        for d in difficulties:
            if preselected_by_diff[d] > per_difficulty:
                print(
                    f"Warning: Difficulty {d} has {preselected_by_diff[d]} preselected questions, "
                    f"exceeding target {per_difficulty}"
                )
            needed = max(per_difficulty - preselected_by_diff[d], 0)
            pool = remaining_by_diff[d]
            n = min(needed, len(pool))
            if n < needed:
                print(
                    f"Warning: Only {len(pool)} remaining questions available for difficulty {d}, "
                    f"requested {needed}"
                )
            if n > 0:
                sampled.extend(random.sample(pool, n))
    elif per_difficulty is not None:
        # Phase 1: Stratified sampling (reuse existing function for guaranteed balance)
        sampled = stratified_sample(questions, per_difficulty, seed, difficulties)
    else:
        raise ValueError("per_difficulty is required unless min_per_template is provided")

    # Phase 2: Regroup by game_id
    by_game: dict[str, list[dict]] = defaultdict(list)
    for q in sampled:
        by_game[q["game_id"]].append(q)

    # Return as list of batches (sorted by game_id for reproducibility)
    return [by_game[gid] for gid in sorted(by_game.keys())]


def sample_questions_by_game(
    questions: list[dict],
    questions_per_game: int,
    num_games: int,
    seed: int,
    difficulties: list[int] | None = None,
) -> list[list[dict]]:
    """
    Sample questions grouped by game for batched evaluation.

    Selects random games, then samples questions from each game with
    balanced difficulty distribution. This enables batched prompts that
    share a single world report across multiple questions.

    Args:
        questions: Full list of questions to sample from
        questions_per_game: Number of questions to sample from each game
        num_games: Number of games to select
        seed: Random seed for reproducibility
        difficulties: Difficulty levels to include (default: [2, 3, 4, 5, 6])

    Returns:
        List of question batches, one list per game. Each batch contains
        up to `questions_per_game` questions from a single game.

    Example:
        >>> questions = load_all_questions(Path("data/questions"))
        >>> batches = sample_questions_by_game(questions, questions_per_game=5, num_games=20, seed=42)
        >>> len(batches)  # 20 games
        20
        >>> len(batches[0])  # 5 questions per game
        5
    """
    if difficulties is None:
        difficulties = [2, 3, 4, 5, 6]

    random.seed(seed)

    # Group questions by game
    by_game: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        game_id = q.get("game_id")
        d = q.get("difficulty", {}).get("composite", 0)
        if d in difficulties:
            by_game[game_id].append(q)

    # Filter games that have enough questions
    eligible_games = [
        game_id for game_id, qs in by_game.items()
        if len(qs) >= questions_per_game
    ]

    if len(eligible_games) < num_games:
        print(f"Warning: Only {len(eligible_games)} games have >= {questions_per_game} questions, "
              f"requested {num_games} games")
        num_games = len(eligible_games)

    # Select random games
    selected_games = random.sample(eligible_games, num_games) if eligible_games else []

    # Sample questions from each game, balancing across difficulties
    batches = []
    for game_id in selected_games:
        game_questions = by_game[game_id]

        # Group by difficulty within this game
        by_difficulty: dict[int, list[dict]] = defaultdict(list)
        for q in game_questions:
            d = q.get("difficulty", {}).get("composite", 0)
            by_difficulty[d].append(q)

        # Sample with difficulty balance
        batch = []
        per_difficulty = questions_per_game // len(difficulties)
        remainder = questions_per_game % len(difficulties)

        for i, d in enumerate(difficulties):
            pool = by_difficulty[d]
            # Add one extra to first `remainder` difficulties to handle uneven division
            n = per_difficulty + (1 if i < remainder else 0)
            n = min(n, len(pool))
            if pool and n > 0:
                batch.extend(random.sample(pool, n))

        # If we didn't get enough due to difficulty imbalance, fill from any difficulty
        if len(batch) < questions_per_game:
            remaining = [q for q in game_questions if q not in batch]
            needed = questions_per_game - len(batch)
            if remaining:
                batch.extend(random.sample(remaining, min(needed, len(remaining))))

        batches.append(batch)

    return batches
