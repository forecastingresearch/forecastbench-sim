#!/usr/bin/env python3
"""
Set up evaluation directories for the lookback conditional experiment.

Creates paired question sets where:
- Baseline: unconditional questions with baseline ground truth
- Lookback: same questions prepended with "Given that {civ} discovers {tech} at turn {T}, ..."
  using events that ACTUALLY HAPPENED in the baseline game

This tests whether conditioning on TRUE future events helps, hurts, or does nothing.

Usage:
    python scripts/setup_lookback_conditional_eval.py
"""

import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Seeds to process (same as other conditional experiments, skip seed3)
SEEDS = ["seed0", "seed1", "seed2", "seed4", "seed5", "seed6", "seed7", "seed8", "seed9", "seed10"]

# Resolution turns used in baseline questions
RESOLUTION_TURNS = [90, 120, 150]


def load_game_data(games_dir: Path, seed: str) -> dict:
    """Load game data for a seed."""
    path = games_dir / f"{seed}_data.json"
    with open(path) as f:
        return json.load(f)


def get_other_player_tech_events(game_data: dict, min_turn: int = 61, max_turn: int = 150) -> list[dict]:
    """Get tech discovery events by non-player-0 civs in the turn range."""
    events = []
    for e in game_data.get("events", []):
        if (
            e.get("type") == "tech_discovered"
            and e.get("player_id", 0) != 0
            and min_turn <= e.get("turn", 0) <= max_turn
        ):
            events.append(e)
    return events


def get_civ_name(civilizations: dict, player_id: int) -> str:
    """Get civilization name from player_id."""
    return civilizations.get(str(player_id), {}).get("name", f"Player {player_id}")


def extract_player_ids_from_question(q: dict) -> set[int]:
    """Extract all player IDs referenced in a question's parameters."""
    params = q.get("parameters", {})
    ids = set()
    for key in ["player_id", "player_id_a", "player_id_b"]:
        val = params.get(key)
        if val is not None:
            ids.add(int(val))
    return ids


def find_best_event_for_question(
    question: dict,
    events_by_player: dict[int, list[dict]],
    all_events: list[dict],
) -> dict | None:
    """Find the best conditioning event for a question.

    Preference:
    1. Latest event before resolution_turn involving a non-player-0 player in the question
    2. If no match, use any event before resolution_turn (for player-0-only questions)
    """
    resolution_turn = question.get("resolution_turn")
    if resolution_turn is None:
        resolution_turn = question.get("parameters", {}).get("resolution_turn")
    if resolution_turn is None:
        return None

    player_ids = extract_player_ids_from_question(question)
    non_zero_ids = player_ids - {0}

    # Try to find event involving a question-relevant player
    if non_zero_ids:
        candidates = []
        for pid in non_zero_ids:
            for e in events_by_player.get(pid, []):
                if e["turn"] < resolution_turn:
                    candidates.append(e)
        if candidates:
            # Pick the latest event before resolution
            return max(candidates, key=lambda e: e["turn"])

    # Fall back to any event before resolution_turn
    candidates = [e for e in all_events if e["turn"] < resolution_turn]
    if candidates:
        return max(candidates, key=lambda e: e["turn"])

    return None


def make_lookback_text(event: dict, original_text: str, civilizations: dict) -> str:
    """Prepend lookback conditioning to question text."""
    civ_name = get_civ_name(civilizations, event["player_id"])
    tech_name = event.get("metadata", {}).get("tech_name", "a technology")
    turn = event["turn"]
    return f"Given that the {civ_name} civilization discovers {tech_name} at turn {turn}, {original_text[0].lower()}{original_text[1:]}"


def setup_lookback_eval():
    base_dir = Path(__file__).parent.parent
    games_dir = base_dir / "data" / "games"
    questions_dir = base_dir / "data" / "questions"

    baseline_eval_dir = base_dir / "data" / "conditional" / "lookback_giventhat" / "baseline"
    conditional_eval_dir = base_dir / "data" / "conditional" / "lookback_giventhat" / "conditional"

    # Load all questions from combined file
    questions_all_path = questions_dir / "questions_all.json"
    print(f"Loading questions from {questions_all_path}...")
    with open(questions_all_path) as f:
        all_data = json.load(f)

    # Group questions by game_id (seed)
    questions_by_seed: dict[str, list[dict]] = defaultdict(list)
    for q in all_data.get("questions", []):
        game_id = q.get("parameters", {}).get("game_id")
        if game_id:
            questions_by_seed[game_id].append(q)
    print(f"Loaded {len(all_data['questions'])} questions across {len(questions_by_seed)} seeds")

    total_baseline = 0
    total_conditional = 0

    print()
    print("Setting up evaluation directories for lookback conditional experiment...")
    print()

    for seed in SEEDS:
        game_data_path = games_dir / f"{seed}_data.json"
        if not game_data_path.exists():
            print(f"  {seed}: Skipping - no game data")
            continue

        world_report_dir = questions_dir / seed / "world_report"
        if not world_report_dir.exists():
            print(f"  {seed}: Skipping - no world_report")
            continue

        if seed not in questions_by_seed:
            print(f"  {seed}: Skipping - no questions in combined file")
            continue

        print(f"  Processing {seed}...")

        game_data = load_game_data(games_dir, seed)
        civilizations = game_data.get("civilizations", {})

        # Get other-player tech events
        events = get_other_player_tech_events(game_data)
        print(f"    Found {len(events)} other-player tech events (turns 61-150)")

        # Index events by player_id
        events_by_player: dict[int, list[dict]] = defaultdict(list)
        for e in events:
            events_by_player[e["player_id"]].append(e)

        seed_questions = questions_by_seed[seed]
        checkpoint_turn = 60

        baseline_questions = []
        conditional_questions = []
        skipped = 0

        for q in seed_questions:
            # Only use questions with resolution turns in our range
            res_turn = q.get("resolution_turn")
            if res_turn is None:
                res_turn = q.get("parameters", {}).get("resolution_turn")
            if res_turn is None or res_turn not in RESOLUTION_TURNS:
                skipped += 1
                continue

            # Find a conditioning event
            event = find_best_event_for_question(q, events_by_player, events)
            if event is None:
                skipped += 1
                continue

            # Determine ground truth
            resolution = q.get("resolution", {})
            question_type = q.get("question_type", "binary")
            if question_type == "continuous":
                gt = resolution.get("value") or resolution.get("value_at_resolution")
                if gt is None:
                    skipped += 1
                    continue
            else:
                answer = resolution.get("answer")
                if answer is None:
                    skipped += 1
                    continue

            # Determine horizon
            diff = res_turn - checkpoint_turn
            if diff <= 30:
                horizon = "H1"
            elif diff <= 60:
                horizon = "H2"
            else:
                horizon = "H3"

            # Build baseline question entry
            base_entry = {
                "question_id": q["question_id"],
                "template_id": q.get("template_id", "unknown"),
                "resolution_turn": res_turn,
                "horizon": horizon,
                "question_type": question_type,
                "parameters": {
                    **q.get("parameters", {}),
                    "checkpoint_turn": checkpoint_turn,
                },
                "question_text": q["question_text"],
                "resolution": resolution,
            }
            baseline_questions.append(base_entry)

            # Build lookback question entry
            lookback_text = make_lookback_text(event, q["question_text"], civilizations)
            cond_entry = {
                "question_id": f"{q['question_id']}_lookback",
                "template_id": f"lookback_{q.get('template_id', 'unknown')}",
                "resolution_turn": res_turn,
                "horizon": horizon,
                "question_type": question_type,
                "parameters": {
                    **q.get("parameters", {}),
                    "checkpoint_turn": checkpoint_turn,
                    "lookback_event_turn": event["turn"],
                    "lookback_event_player": event["player_id"],
                    "lookback_event_tech": event.get("metadata", {}).get("tech_name", "unknown"),
                },
                "question_text": lookback_text,
                "resolution": resolution,  # Same ground truth as baseline
            }
            conditional_questions.append(cond_entry)

        n_baseline = len(baseline_questions)
        n_conditional = len(conditional_questions)
        total_baseline += n_baseline
        total_conditional += n_conditional

        print(f"    Paired: {n_baseline} questions, skipped: {skipped}")

        # Write baseline eval dir
        baseline_seed_dir = baseline_eval_dir / seed
        baseline_seed_dir.mkdir(parents=True, exist_ok=True)
        with open(baseline_seed_dir / "questions.json", "w") as f:
            json.dump({
                "game_id": seed,
                "snapshot_turn": checkpoint_turn,
                "generated_at": datetime.now().isoformat() + "Z",
                "civilizations": civilizations,
                "questions": baseline_questions,
                "condition_metadata": {
                    "condition_type": "baseline",
                    "experiment": "lookback_conditional",
                },
            }, f, indent=2)
        if (baseline_seed_dir / "world_report").exists():
            shutil.rmtree(baseline_seed_dir / "world_report")
        shutil.copytree(world_report_dir, baseline_seed_dir / "world_report")

        # Write conditional eval dir
        conditional_seed_dir = conditional_eval_dir / seed
        conditional_seed_dir.mkdir(parents=True, exist_ok=True)
        with open(conditional_seed_dir / "questions.json", "w") as f:
            json.dump({
                "game_id": seed,
                "snapshot_turn": checkpoint_turn,
                "generated_at": datetime.now().isoformat() + "Z",
                "civilizations": civilizations,
                "questions": conditional_questions,
                "condition_metadata": {
                    "condition_type": "lookback",
                    "experiment": "lookback_conditional",
                    "description": "Conditioning on true future events (other players' tech discoveries)",
                },
            }, f, indent=2)
        if (conditional_seed_dir / "world_report").exists():
            shutil.rmtree(conditional_seed_dir / "world_report")
        shutil.copytree(world_report_dir, conditional_seed_dir / "world_report")

    print()
    print("=" * 60)
    print(f"  Baseline questions:    {total_baseline}")
    print(f"  Conditional questions: {total_conditional}")
    print()
    print("Evaluation directories:")
    print(f"  {baseline_eval_dir}")
    print(f"  {conditional_eval_dir}")
    print()
    print("To run evaluations:")
    print(f"  uv run python scripts/evaluate_llm_forecasts_parallel.py \\")
    print(f"    --data-dir data/conditional/lookback/baseline --all \\")
    print(f"    --models openai/o3-2025-04-16 anthropic/claude-opus-4-5-20251101 openai/gpt-4.1-2025-04-14 \\")
    print(f"    -o data/results/lookback_baseline_eval.json")
    print()
    print(f"  uv run python scripts/evaluate_llm_forecasts_parallel.py \\")
    print(f"    --data-dir data/conditional/lookback/conditional --all \\")
    print(f"    --models openai/o3-2025-04-16 anthropic/claude-opus-4-5-20251101 openai/gpt-4.1-2025-04-14 \\")
    print(f"    -o data/results/lookback_conditional_eval.json")


if __name__ == "__main__":
    setup_lookback_eval()
