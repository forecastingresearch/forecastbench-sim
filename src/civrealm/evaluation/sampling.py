"""Stratified sampling for CivBench evaluation questions."""

import json
import random
from collections import defaultdict
from pathlib import Path


def load_all_questions(
    data_dir: Path,
    template_filter: str | None = None,
    include_h0: bool = True,
    min_difficulty: float | None = None,
    max_difficulty: float | None = None,
    difficulty_percentile_min: float | None = None,
    difficulty_percentile_max: float | None = None,
) -> list[dict]:
    """
    Load all questions from game folders or a combined questions file.

    Supports two formats:
    1. Combined file: data_dir/questions_all.json (from generate_questions_batch.py)
    2. Per-game directories: data_dir/{game_id}/questions.json

    Args:
        data_dir: Directory containing questions (either combined file or game folders)
        template_filter: Optional filter to only load questions of a specific template
        include_h0: Whether to also load from h0_questions.json (default: True)
        min_difficulty: Minimum difficulty score (0.0-1.0, higher = harder)
        max_difficulty: Maximum difficulty score (0.0-1.0, higher = harder)
        difficulty_percentile_min: Minimum difficulty percentile (0-100)
        difficulty_percentile_max: Maximum difficulty percentile (0-100)

    Returns:
        List of question dicts with keys: game_id, question_id, question_text,
        ground_truth, parameters, template_id, horizon, empirical_difficulty (if present)
    """
    data_dir = Path(data_dir)
    questions = []

    # Check for combined questions file first (from generate_questions_batch.py)
    combined_file = data_dir / "questions_all.json"
    if combined_file.exists():
        questions = _load_from_combined_file(combined_file, template_filter)
    else:
        # Fall back to per-game directory structure
        questions = _load_from_game_directories(data_dir, template_filter, include_h0)

    # Apply difficulty filtering if specified
    if any([min_difficulty, max_difficulty, difficulty_percentile_min, difficulty_percentile_max]):
        questions = _filter_by_difficulty(
            questions,
            min_difficulty=min_difficulty,
            max_difficulty=max_difficulty,
            percentile_min=difficulty_percentile_min,
            percentile_max=difficulty_percentile_max,
        )

    return questions


def _load_from_combined_file(
    combined_file: Path,
    template_filter: str | None = None,
) -> list[dict]:
    """Load questions from a combined questions_all.json file."""
    questions = []
    missing_game_id_warned = False

    with open(combined_file) as f:
        data = json.load(f)

    # Combined file has metadata and questions at top level
    for q in data.get("questions", []):
        if template_filter and q.get("template_id") != template_filter:
            continue

        resolution = q.get("resolution", {})
        answer = resolution.get("answer")
        if answer is None:
            continue

        # Handle both new format (horizon at top level) and old format (in difficulty)
        horizon = q.get("horizon")
        if horizon is None:
            difficulty = q.get("difficulty", {})
            horizon = difficulty.get("horizon", "H1")

        # Get empirical difficulty if present
        empirical_difficulty = q.get("empirical_difficulty")

        # game_id is stored in parameters for combined format
        game_id = q.get("parameters", {}).get("game_id")
        if game_id is None:
            if not missing_game_id_warned:
                print("Warning: questions_all.json is missing game_id in parameters. "
                      "Re-run generate_questions_batch.py to fix. Using 'unknown' as fallback.")
                missing_game_id_warned = True
            game_id = "unknown"

        questions.append({
            "game_id": game_id,
            "question_id": q.get("question_id"),
            "template_id": q.get("template_id"),
            "question_text": q.get("question_text"),
            "ground_truth": bool(answer),
            "parameters": q.get("parameters", {}),
            "horizon": horizon,
            "empirical_difficulty": empirical_difficulty,
        })

    return questions


def _load_from_game_directories(
    data_dir: Path,
    template_filter: str | None = None,
    include_h0: bool = True,
) -> list[dict]:
    """Load questions from per-game directory structure."""
    questions = []

    # Question files to load from (in order of priority)
    question_files = ["questions.json"]
    if include_h0:
        question_files.append("h0_questions.json")

    for game_dir in sorted(data_dir.iterdir()):
        if not game_dir.is_dir():
            continue

        for qfile_name in question_files:
            questions_file = game_dir / qfile_name
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

                # Handle both new format (horizon at top level) and old format (in difficulty)
                horizon = q.get("horizon")
                if horizon is None:
                    difficulty = q.get("difficulty", {})
                    horizon = difficulty.get("horizon", "H1")

                # Get empirical difficulty if present
                empirical_difficulty = q.get("empirical_difficulty")

                questions.append({
                    "game_id": game_id,
                    "question_id": q.get("question_id"),
                    "template_id": q.get("template_id"),
                    "question_text": q.get("question_text"),
                    "ground_truth": bool(answer),
                    "parameters": q.get("parameters", {}),
                    "horizon": horizon,
                    "empirical_difficulty": empirical_difficulty,
                })

    return questions


def _filter_by_difficulty(
    questions: list[dict],
    min_difficulty: float | None = None,
    max_difficulty: float | None = None,
    percentile_min: float | None = None,
    percentile_max: float | None = None,
) -> list[dict]:
    """
    Filter questions by difficulty score or percentile.

    Questions without empirical_difficulty are excluded when filtering.

    Args:
        questions: List of question dicts
        min_difficulty: Minimum difficulty score (0.0-1.0)
        max_difficulty: Maximum difficulty score (0.0-1.0)
        percentile_min: Minimum difficulty percentile (0-100)
        percentile_max: Maximum difficulty percentile (0-100)

    Returns:
        Filtered list of questions
    """
    filtered = []

    for q in questions:
        emp_diff = q.get("empirical_difficulty")

        # Skip questions without difficulty data when filtering
        if emp_diff is None:
            continue

        score = emp_diff.get("score")
        percentile = emp_diff.get("percentile")

        # Check score bounds
        if min_difficulty is not None and (score is None or score < min_difficulty):
            continue
        if max_difficulty is not None and (score is None or score > max_difficulty):
            continue

        # Check percentile bounds
        if percentile_min is not None and (percentile is None or percentile < percentile_min):
            continue
        if percentile_max is not None and (percentile is None or percentile > percentile_max):
            continue

        filtered.append(q)

    return filtered


def stratified_sample_by_horizon_template(
    questions: list[dict],
    per_pair: int,
    seed: int,
    horizons: list[str] | None = None,
    templates: list[str] | None = None,
) -> list[dict]:
    """
    Sample questions with balanced representation across (horizon, template) pairs.

    Guarantees up to `per_pair` questions from each available (horizon, template)
    combination, providing fine-grained control for anchor runs and difficulty
    calibration. Templates are selected per horizon from what's available.

    Args:
        questions: Full list of questions to sample from
        per_pair: Number of questions to sample from each (horizon, template) pair
        seed: Random seed for reproducibility
        horizons: Horizon levels to include (default: all horizons found)
        templates: Template IDs to include (default: all templates found)

    Returns:
        List of sampled questions, balanced across (horizon, template) pairs.

    Example:
        >>> questions = load_all_questions(Path("data/questions"))
        >>> sampled = stratified_sample_by_horizon_template(questions, per_pair=1, seed=42)
        >>> # With 13 templates and 8 horizons, up to 104 questions (1 per pair)
    """
    random.seed(seed)
    sampled = []

    # Group questions by (horizon, template_id)
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for q in questions:
        horizon = q.get("horizon", "H1")
        template_id = q.get("template_id", "unknown")
        by_pair[(horizon, template_id)].append(q)

    # Determine which horizons to sample from
    all_horizons = set(h for h, _ in by_pair.keys())
    if horizons is None:
        horizons = sorted(all_horizons)

    template_filter = set(templates) if templates is not None else None

    # Sample from each (horizon, template) pair (use all templates available per horizon)
    for horizon in horizons:
        templates_for_horizon = sorted(
            t for (h, t) in by_pair.keys()
            if h == horizon and (template_filter is None or t in template_filter)
        )
        for template_id in templates_for_horizon:
            pool = by_pair.get((horizon, template_id), [])
            n = min(per_pair, len(pool))

            if n < per_pair and pool:
                print(f"Warning: Only {len(pool)} questions available for "
                      f"({horizon}, {template_id}), requested {per_pair}")

            if pool:
                sampled.extend(random.sample(pool, n))

    return sampled


def stratified_sample(
    questions: list[dict],
    per_template: int,
    seed: int,
    templates: list[str] | None = None,
) -> list[dict]:
    """
    Sample questions with balanced representation across question templates.

    Guarantees exactly `per_template` questions from each template,
    resulting in balanced samples across question types.

    Args:
        questions: Full list of questions to sample from
        per_template: Number of questions to sample from each template
        seed: Random seed for reproducibility
        templates: Template IDs to sample from (default: all templates found)

    Returns:
        List of sampled questions, balanced across templates.
        Total size = per_template * len(templates)

    Example:
        >>> questions = load_all_questions(Path("data/questions"))
        >>> sampled = stratified_sample(questions, per_template=10, seed=42)
        >>> len(sampled)  # 10 * num_templates
    """
    random.seed(seed)
    sampled = []

    # Group questions by template_id
    by_template: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        template_id = q.get("template_id", "unknown")
        by_template[template_id].append(q)

    # Determine which templates to sample from
    if templates is None:
        templates = sorted(by_template.keys())

    # Sample from each template
    for template_id in templates:
        pool = by_template.get(template_id, [])
        n = min(per_template, len(pool))

        if n < per_template:
            print(f"Warning: Only {len(pool)} questions available for template {template_id}, "
                  f"requested {per_template}")

        if pool:
            sampled.extend(random.sample(pool, n))

    return sampled


def get_template_distribution(questions: list[dict]) -> dict[str, int]:
    """
    Get the count of questions for each template.

    Args:
        questions: List of question dicts

    Returns:
        Dict mapping template_id to count
    """
    dist: dict[str, int] = defaultdict(int)
    for q in questions:
        template_id = q.get("template_id", "unknown")
        dist[template_id] += 1
    return dict(sorted(dist.items()))


def get_horizon_template_distribution(questions: list[dict]) -> dict[str, dict[str, int]]:
    """
    Get the count of questions for each (horizon, template) pair.

    Args:
        questions: List of question dicts

    Returns:
        Nested dict: {horizon: {template_id: count}}
    """
    dist: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for q in questions:
        horizon = q.get("horizon", "H1")
        template_id = q.get("template_id", "unknown")
        dist[horizon][template_id] += 1

    # Convert to regular dicts and sort
    return {
        h: dict(sorted(templates.items()))
        for h, templates in sorted(dist.items())
    }


def stratified_sample_batched(
    questions: list[dict],
    per_template: int,
    seed: int,
    templates: list[str] | None = None,
) -> list[list[dict]]:
    """
    Sample questions with balanced templates, then group by game for batching.

    Two-phase approach:
    1. Stratified sample: exactly `per_template` questions from each template
    2. Regroup by game_id: organize sampled questions into batches by game

    This gives both:
    - Guaranteed template balance (exactly N per template)
    - Token efficiency (questions from same game share one world report)

    Args:
        questions: Full list of questions to sample from
        per_template: Number of questions per template
        seed: Random seed for reproducibility
        templates: Template IDs to sample (default: all templates found)

    Returns:
        List of question batches, grouped by game_id.
        Total questions = per_template * len(templates)

    Example:
        >>> batches = stratified_sample_batched(questions, per_template=10, seed=42)
        >>> sum(len(b) for b in batches)  # 10 * num_templates total
    """
    # Phase 1: Stratified sampling (reuse existing function for guaranteed balance)
    sampled = stratified_sample(questions, per_template, seed, templates)

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
    templates: list[str] | None = None,
) -> list[list[dict]]:
    """
    Sample questions grouped by game for batched evaluation.

    Selects random games, then samples questions from each game with
    balanced template distribution. This enables batched prompts that
    share a single world report across multiple questions.

    Args:
        questions: Full list of questions to sample from
        questions_per_game: Number of questions to sample from each game
        num_games: Number of games to select
        seed: Random seed for reproducibility
        templates: Template IDs to include (default: all templates found)

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
    random.seed(seed)

    # Group questions by game, optionally filtering by template
    by_game: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        template_id = q.get("template_id", "unknown")
        if templates is None or template_id in templates:
            by_game[q.get("game_id")].append(q)

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

    # Sample questions from each game, balancing across templates
    batches = []
    for game_id in selected_games:
        game_questions = by_game[game_id]

        # Group by template within this game
        by_template: dict[str, list[dict]] = defaultdict(list)
        for q in game_questions:
            template_id = q.get("template_id", "unknown")
            by_template[template_id].append(q)

        # Get list of templates available in this game
        available_templates = sorted(by_template.keys())

        # Sample with template balance
        batch = []
        per_template = questions_per_game // len(available_templates) if available_templates else 0
        remainder = questions_per_game % len(available_templates) if available_templates else 0

        for i, template_id in enumerate(available_templates):
            pool = by_template[template_id]
            # Add one extra to first `remainder` templates to handle uneven division
            n = per_template + (1 if i < remainder else 0)
            n = min(n, len(pool))
            if pool and n > 0:
                batch.extend(random.sample(pool, n))

        # If we didn't get enough due to template imbalance, fill from any template
        if len(batch) < questions_per_game:
            remaining = [q for q in game_questions if q not in batch]
            needed = questions_per_game - len(batch)
            if remaining:
                batch.extend(random.sample(remaining, min(needed, len(remaining))))

        batches.append(batch)

    return batches
