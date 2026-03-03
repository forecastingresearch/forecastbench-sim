#!/usr/bin/env python3
"""Generate conditional_results.json from existing fork savegames.

This script generates and resolves conditional forecasting questions by comparing
baseline (from game_data.json) with fork (from savegames).

Uses the same QuestionResolver as unconditional questions for consistency.

Usage:
    uv run python scripts/generate_conditional_results.py \
        --baseline-dir logs/recordings/seed0 \
        --fork-dir logs/recordings/seed0forkgoldadd5000p0 \
        --condition "gold_add:0:5000" \
        --checkpoint-turn 60 \
        --end-turn 300 \
        --game-data data/games/seed0_data.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Parse arguments before importing civrealm
parser = argparse.ArgumentParser(
    description="Generate conditional_results.json from existing fork savegames"
)
parser.add_argument("--baseline-dir", required=True, help="Path to baseline recording directory")
parser.add_argument("--fork-dir", required=True, help="Path to fork recording directory")
parser.add_argument("--condition", required=True, help="Condition in format 'type:player_id:value' (e.g., 'gold_add:0:5000')")
parser.add_argument("--checkpoint-turn", type=int, default=60, help="Turn the fork started from (default: 60)")
parser.add_argument("--end-turn", type=int, required=True, help="Turn to evaluate at")
parser.add_argument("--game-data", type=str, required=True, help="Path to baseline game data JSON (required)")
parser.add_argument("--output", type=str, default=None, help="Output directory (default: fork-dir)")
parser.add_argument("--questions-from", type=str, default=None,
                    help="Load question definitions from an existing questions.json instead of generating independently. "
                         "Resolves those questions against the fork game state.")
parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed progress")
args = parser.parse_args()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from civrealm.world_reports.utils.savegame_parser import (
    find_local_savegame_for_turn,
    load_local_savegame,
    decompress_savegame_content,
    parse_player_states_for_conditional,
    parse_player_technologies,
    parse_city_wonders,
)
from civrealm.world_reports.questions import (
    ConditionalQuestion,
    ConditionalQuestionBank,
    ConditionalQuestionGenerator,
    ConditionalResult,
    ForkOutcome,
    create_condition,
    save_conditional_bank,
)
from civrealm.world_reports.questions.resolver import QuestionResolver
from civrealm.world_reports.questions.schema import QuestionInstance


def parse_condition_string(cond_str: str) -> dict:
    """Parse condition string 'type:player_id:value' into dict."""
    parts = cond_str.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid condition format: {cond_str}. Expected 'type:player_id:value'")

    cond_type = parts[0]
    player_id = int(parts[1])

    if cond_type in ("gold", "gold_add"):
        value = int(parts[2])
    elif cond_type == "government":
        value = parts[2]
    elif cond_type == "tech":
        value = int(parts[2])
    else:
        raise ValueError(f"Unknown condition type: {cond_type}")

    return {"type": cond_type, "player_id": player_id, "value": value}


def get_username_from_dir(recording_dir: str) -> str:
    """Extract username from recording directory name."""
    return Path(recording_dir).name


def load_savegame_content(recording_dir: str, turn: int, verbose: bool = False) -> str | None:
    """Load and decompress savegame content at the given turn."""
    username = get_username_from_dir(recording_dir)

    savegame_name = find_local_savegame_for_turn(username, turn, recording_dir)
    if not savegame_name:
        if verbose:
            print(f"  Warning: No savegame found for turn {turn} in {recording_dir}")
        return None

    result = load_local_savegame(savegame_name, recording_dir)
    if not result:
        if verbose:
            print(f"  Warning: Could not load savegame {savegame_name}")
        return None

    savegame_bytes, actual_filename = result

    try:
        content = decompress_savegame_content(savegame_bytes, actual_filename)
        if verbose:
            print(f"  Loaded savegame from {actual_filename}")
        return content
    except Exception as e:
        if verbose:
            print(f"  Warning: Error parsing savegame: {e}")
        return None


def build_game_data_from_savegames(
    recording_dir: str,
    turns: list[int],
    base_game_data: dict,
    verbose: bool = False,
) -> dict:
    """Build a game_data-like structure from savegames for use with QuestionResolver.

    Args:
        recording_dir: Path to recording directory with savegames
        turns: List of turns to load data for
        base_game_data: Base game_data to copy civilizations and metadata from
        verbose: Print progress

    Returns:
        Dict compatible with QuestionResolver (has time_series, events, snapshots)
    """
    # Start with civilizations and metadata from base
    game_data = {
        "metadata": base_game_data.get("metadata", {}),
        "civilizations": base_game_data.get("civilizations", {}),
        "time_series": {},
        "events": [],
        "snapshots": {},
    }

    # Load savegame data for each turn
    for turn in turns:
        content = load_savegame_content(recording_dir, turn, verbose)
        if content is None:
            continue

        # Parse player states
        player_states = parse_player_states_for_conditional(content)

        # Build time_series entries for this turn
        for metric in ["treasury", "population", "scores", "territory_size", "techs_known", "cities_count"]:
            if metric not in game_data["time_series"]:
                game_data["time_series"][metric] = {}
            game_data["time_series"][metric][str(turn)] = {}

        for pid, state in player_states.items():
            pid_str = str(pid)
            game_data["time_series"]["treasury"][str(turn)][pid_str] = state.get("gold", 0)
            # Population = sum of citizen types (matches game-state aggregate_city_metric('size'))
            game_data["time_series"]["population"][str(turn)][pid_str] = state.get("citizen_population", 0)
            # Territory = owned tile count from map (matches game-state calculate_territory_size)
            game_data["time_series"]["territory_size"][str(turn)][pid_str] = state.get("tile_count", 0)
            # Score = actual Freeciv score from score section
            game_data["time_series"]["scores"][str(turn)][pid_str] = state.get("total_score", 0)
            game_data["time_series"]["techs_known"][str(turn)][pid_str] = state.get("techs", 0)
            game_data["time_series"]["cities_count"][str(turn)][pid_str] = state.get("cities", 0)

        # Build snapshots for this turn
        game_data["snapshots"][str(turn)] = {
            "scores": {str(pid): state.get("total_score", 0)
                       for pid, state in player_states.items()},
            "tech_count": {str(pid): state.get("techs", 0) for pid, state in player_states.items()},
            "city_count": {str(pid): state.get("cities", 0) for pid, state in player_states.items()},
        }

        # Parse detailed tech info for tech_discovered questions
        try:
            player_techs = parse_player_technologies(content)
            for pid, tech_ids in player_techs.items():
                for tech_id in tech_ids:
                    # Create synthetic tech_discovered event
                    game_data["events"].append({
                        "turn": turn,  # We don't know exact turn, use this turn
                        "type": "tech_discovered",
                        "player_id": pid,
                        "metadata": {"tech_id": str(tech_id), "tech_name": f"Tech #{tech_id}"},
                    })
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not parse technologies: {e}")

        # Parse wonder info for wonder_completed questions
        try:
            wonder_data = parse_city_wonders(content)
            for pid, pdata in wonder_data.items():
                for imp_id in pdata.get("improvements_built", set()):
                    # Note: We can't distinguish wonders from regular improvements without ruleset
                    # For now, skip wonder events (they'll be unresolvable)
                    pass
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not parse wonders: {e}")

    return game_data


def conditional_to_question_instance(cond_question, snapshot_turn: int) -> QuestionInstance:
    """Convert a ConditionalQuestion to a QuestionInstance for the resolver."""
    return QuestionInstance(
        question_id=cond_question.conditional_id,
        template_id=cond_question.target_template_id,
        resolution_turn=cond_question.resolution_turn,
        horizon="H1",  # Not used by resolver
        parameters=cond_question.target_parameters,
        question_text="",  # Not used by resolver
        resolution=None,
    )


def main():
    baseline_dir = Path(args.baseline_dir)
    fork_dir = Path(args.fork_dir)

    if not baseline_dir.exists():
        print(f"Error: Baseline directory not found: {baseline_dir}")
        return 1

    if not fork_dir.exists():
        print(f"Error: Fork directory not found: {fork_dir}")
        return 1

    # Load game data (required for baseline resolution)
    game_data_path = Path(args.game_data)
    if not game_data_path.exists():
        print(f"Error: Game data file not found: {game_data_path}")
        return 1

    with open(game_data_path) as f:
        game_data = json.load(f)

    # Parse condition
    cond_dict = parse_condition_string(args.condition)

    print(f"Generating conditional results")
    print(f"  Baseline: {baseline_dir}")
    print(f"  Fork: {fork_dir}")
    print(f"  Condition: {args.condition}")
    print(f"  Checkpoint turn: {args.checkpoint_turn}")
    print(f"  End turn: {args.end_turn}")
    print(f"  Game data: {game_data_path}")

    # Infer game_id from baseline directory
    game_id = baseline_dir.name

    # Calculate resolution turns (H1-H6 from checkpoint)
    horizons = [30, 60, 90, 120, 150, 180]
    resolution_turns = [args.checkpoint_turn + h for h in horizons if args.checkpoint_turn + h <= args.end_turn]
    print(f"  Resolution turns: {resolution_turns}")

    # Build fork game_data from savegames
    print(f"\nBuilding fork game_data from savegames...")
    fork_game_data = build_game_data_from_savegames(
        str(fork_dir), resolution_turns, game_data, args.verbose
    )

    # Check which turns have data
    valid_turns = [t for t in resolution_turns if str(t) in fork_game_data["snapshots"]]
    if not valid_turns:
        print("Error: No valid resolution turns found in fork savegames")
        return 1
    print(f"  Valid resolution turns: {valid_turns}")

    # Get civ name
    civ_data = game_data.get("civilizations", {}).get(str(cond_dict["player_id"]), {})
    civ_name = civ_data.get("name", f"Player {cond_dict['player_id']}")

    # Create condition object
    condition = create_condition(
        condition_type=cond_dict["type"],
        player_id=cond_dict["player_id"],
        value=cond_dict["value"],
        civ_name=civ_name,
    )

    # Generate or load conditional questions
    if args.questions_from:
        # Load question definitions from existing questions.json
        questions_from_path = Path(args.questions_from)
        if not questions_from_path.exists():
            print(f"Error: --questions-from file not found: {questions_from_path}")
            return 1

        print(f"\nLoading questions from {questions_from_path}...")
        with open(questions_from_path) as f:
            source_bank = json.load(f)

        # Convert source questions to ConditionalQuestion objects
        source_questions = source_bank.get("questions", [])
        cond_questions = []
        for sq in source_questions:
            res_turn = sq["resolution_turn"]
            if res_turn not in valid_turns:
                continue
            cond_questions.append(ConditionalQuestion(
                conditional_id=sq["question_id"],
                condition=condition,
                target_template_id=sq["template_id"],
                target_parameters=sq["parameters"],
                checkpoint_turn=args.checkpoint_turn,
                resolution_turn=res_turn,
            ))

        cond_bank = ConditionalQuestionBank(
            game_id=game_id,
            checkpoint_turn=args.checkpoint_turn,
            end_turn=args.end_turn,
            conditions=[condition],
            questions=cond_questions,
            results={},
            generated_at=datetime.now().isoformat() + "Z",
        )
        print(f"  Loaded {len(cond_questions)} questions (filtered to valid turns)")
    else:
        print(f"\nGenerating conditional questions...")
        generator = ConditionalQuestionGenerator()

        cond_bank = generator.generate_conditional_bank(
            game_id=game_id,
            game_data=game_data,
            checkpoint_turn=args.checkpoint_turn,
            end_turn=args.end_turn,
            resolution_turns=valid_turns,
            conditions=[condition],
            target_templates=None,  # Use all templates from CONDITION_TARGET_MAP
        )

    print(f"  Generated {len(cond_bank.questions)} conditional questions")
    print(f"  Resolution turns: {sorted(set(q.resolution_turn for q in cond_bank.questions))}")
    print(f"  Templates: {sorted(set(q.target_template_id for q in cond_bank.questions))}")

    # Use QuestionResolver for both baseline and fork
    resolver = QuestionResolver()

    print(f"\nResolving questions using QuestionResolver...")

    effects = []
    resolved_count = 0
    unresolved_templates = set()

    for question in cond_bank.questions:
        resolution_turn = question.resolution_turn

        # Convert to QuestionInstance for resolver
        q_instance = conditional_to_question_instance(question, args.checkpoint_turn)

        # Determine if this is a continuous template
        is_continuous = question.target_template_id.endswith("_continuous")

        # Resolve using baseline game_data
        try:
            baseline_resolution = resolver.resolve(q_instance, game_data, args.checkpoint_turn)
            if is_continuous:
                answer_control = baseline_resolution.value_at_resolution
            else:
                answer_control = baseline_resolution.answer
        except Exception as e:
            if args.verbose:
                print(f"  Warning: Could not resolve baseline for {question.conditional_id}: {e}")
            answer_control = None

        # Resolve using fork game_data (built from savegames)
        try:
            fork_resolution = resolver.resolve(q_instance, fork_game_data, args.checkpoint_turn)
            if is_continuous:
                answer_intervention = fork_resolution.value_at_resolution
            else:
                answer_intervention = fork_resolution.answer
        except Exception as e:
            if args.verbose:
                print(f"  Warning: Could not resolve fork for {question.conditional_id}: {e}")
            answer_intervention = None

        conditional_effect = None
        if answer_control is not None and answer_intervention is not None:
            if isinstance(answer_control, bool) and isinstance(answer_intervention, bool):
                conditional_effect = 1.0 if answer_control != answer_intervention else 0.0
            else:
                conditional_effect = answer_intervention - answer_control
            effects.append(conditional_effect)
            resolved_count += 1
        else:
            unresolved_templates.add(question.target_template_id)

        # Create placeholder outcomes (we don't have full player_states in new format)
        baseline_outcome = ForkOutcome(
            fork_name="baseline",
            success=True,
            final_turn=resolution_turn,
            player_states={},
            error=None,
        )

        intervention_outcome = ForkOutcome(
            fork_name="fork_intervention",
            success=True,
            final_turn=resolution_turn,
            player_states={},
            error=None,
        )

        cond_bank.results[question.conditional_id] = ConditionalResult(
            conditional_id=question.conditional_id,
            control_outcome=baseline_outcome,
            intervention_outcome=intervention_outcome,
            answer_control=answer_control,
            answer_intervention=answer_intervention,
            conditional_effect=conditional_effect,
            computed_at=datetime.now().isoformat() + "Z",
        )

        if args.verbose:
            print(f"  {question.conditional_id} (T{resolution_turn}): control={answer_control}, intervention={answer_intervention}, effect={conditional_effect}")

    # Summary
    print(f"\nResults summary:")
    print(f"  Total questions: {len(cond_bank.questions)}")
    print(f"  Resolved: {resolved_count}")
    if unresolved_templates:
        print(f"  Unresolved templates: {sorted(unresolved_templates)}")
    if effects:
        differing = sum(1 for e in effects if e > 0)
        avg_effect = sum(effects) / len(effects)
        print(f"  Questions with conditional effect: {differing}/{len(effects)} ({100*differing/len(effects):.1f}%)")
        print(f"  Average conditional effect: {avg_effect:.3f}")

    # Save results
    output_dir = Path(args.output) if args.output else fork_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "conditional_results.json"

    save_conditional_bank(cond_bank, results_path)
    print(f"\nSaved conditional results to {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
