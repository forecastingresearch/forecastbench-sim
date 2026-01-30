#!/usr/bin/env python3
"""Generate conditional_results.json from existing fork savegames.

This script enables extracting conditional forecasting results from already-completed
fork runs without re-running the simulation. It parses player states from both the
baseline (original recording) and intervention (fork) savegames, then generates
and resolves conditional questions.

Usage:
    # Basic usage with existing fork
    uv run python scripts/generate_conditional_results.py \
        --baseline-dir logs/recordings/seed0 \
        --fork-dir logs/recordings/seed0forkgoldadd5000p0 \
        --condition "gold_add:0:5000" \
        --end-turn 300

    # Specify output directory
    uv run python scripts/generate_conditional_results.py \
        --baseline-dir logs/recordings/seed0 \
        --fork-dir logs/recordings/seed0forkgoldadd5000p0 \
        --condition "gold_add:0:5000" \
        --end-turn 300 \
        --output data/questions/seed0/

    # With checkpoint turn (for question generation)
    uv run python scripts/generate_conditional_results.py \
        --baseline-dir logs/recordings/seed0 \
        --fork-dir logs/recordings/seed0forkgoldadd5000p0 \
        --condition "gold_add:0:5000" \
        --checkpoint-turn 50 \
        --end-turn 300
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
parser.add_argument("--checkpoint-turn", type=int, default=50, help="Turn the fork started from (default: 50)")
parser.add_argument("--end-turn", type=int, required=True, help="Turn to evaluate at")
parser.add_argument("--game-data", type=str, default=None, help="Path to game data JSON (optional, improves civ names)")
parser.add_argument("--output", type=str, default=None, help="Output directory (default: fork-dir)")
parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed progress")
args = parser.parse_args()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from civrealm.world_reports.utils.savegame_parser import (
    find_local_savegame_for_turn,
    load_local_savegame,
    decompress_savegame_content,
    parse_player_states_for_conditional,
)
from civrealm.world_reports.questions import (
    ConditionalQuestionGenerator,
    ConditionalQuestionBank,
    ConditionalQuestion,
    ConditionalResult,
    Condition,
    ForkOutcome,
    create_condition,
    save_conditional_bank,
)


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

    return {
        "type": cond_type,
        "player_id": player_id,
        "value": value,
    }


def get_username_from_dir(recording_dir: str) -> str:
    """Extract username from recording directory name."""
    return Path(recording_dir).name


def load_player_states(recording_dir: str, turn: int, verbose: bool = False) -> dict[int, dict] | None:
    """Load player states from a savegame at the given turn."""
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
        player_states = parse_player_states_for_conditional(content)
        if verbose:
            print(f"  Loaded player states from {actual_filename}")
        return player_states
    except Exception as e:
        if verbose:
            print(f"  Warning: Error parsing savegame: {e}")
        return None


def resolve_comparative(params: dict, player_states: dict[int, dict], field: str) -> bool | None:
    """Resolve a comparative question (A > B)."""
    player_a = params.get("player_id_a")
    player_b = params.get("player_id_b")

    if player_a is None or player_b is None:
        return None

    state_a = player_states.get(player_a, {})
    state_b = player_states.get(player_b, {})

    value_a = state_a.get(field)
    value_b = state_b.get(field)

    if value_a is None or value_b is None:
        return None

    return value_a > value_b


def resolve_target_question(question: ConditionalQuestion, player_states: dict[int, dict]) -> bool | None:
    """Resolve target question using player states."""
    template_id = question.target_template_id
    params = question.target_parameters

    if template_id == "treasury_comparative":
        return resolve_comparative(params, player_states, "gold")
    elif template_id == "score_comparative":
        # Score not directly in savegame, use proxy
        return resolve_comparative_with_score_proxy(params, player_states)
    elif template_id == "tech_comparative":
        return resolve_comparative(params, player_states, "techs")
    elif template_id == "population_comparative":
        return resolve_comparative(params, player_states, "population")
    elif template_id in ["cities_comparative", "city_count_comparative"]:
        return resolve_comparative(params, player_states, "cities")
    elif template_id == "territory_comparative":
        # Territory uses land_area if available, otherwise fall back to cities as proxy
        return resolve_comparative(params, player_states, "land_area") or \
               resolve_comparative(params, player_states, "cities")
    elif template_id == "score_rank_1":
        # Check if player is ranked #1 by score proxy
        player_id = params.get("player_id")
        if player_id is None:
            return None
        # Compute score proxy for all players and check if this one is ranked #1
        scores = []
        for pid, state in player_states.items():
            score = (
                state.get("techs", 0) * 10 +
                state.get("cities", 0) * 5 +
                state.get("population", 0) // 100 +
                state.get("wonders", 0) * 20
            )
            scores.append((pid, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0] == player_id if scores else None

    return None


def resolve_comparative_with_score_proxy(params: dict, player_states: dict[int, dict]) -> bool | None:
    """Resolve comparative question using a score proxy computed from components."""
    player_a = params.get("player_id_a")
    player_b = params.get("player_id_b")

    if player_a is None or player_b is None:
        return None

    state_a = player_states.get(player_a, {})
    state_b = player_states.get(player_b, {})

    def compute_score_proxy(state: dict) -> int:
        return (
            state.get("techs", 0) * 10 +
            state.get("cities", 0) * 5 +
            state.get("population", 0) // 100 +
            state.get("wonders", 0) * 20
        )

    score_a = compute_score_proxy(state_a)
    score_b = compute_score_proxy(state_b)

    return score_a > score_b


def main():
    baseline_dir = Path(args.baseline_dir)
    fork_dir = Path(args.fork_dir)

    if not baseline_dir.exists():
        print(f"Error: Baseline directory not found: {baseline_dir}")
        return 1

    if not fork_dir.exists():
        print(f"Error: Fork directory not found: {fork_dir}")
        return 1

    # Parse condition
    cond_dict = parse_condition_string(args.condition)

    print(f"Generating conditional results")
    print(f"  Baseline: {baseline_dir}")
    print(f"  Fork: {fork_dir}")
    print(f"  Condition: {args.condition}")
    print(f"  Checkpoint turn: {args.checkpoint_turn}")
    print(f"  End turn: {args.end_turn}")

    # Load game data if provided (for better civ names)
    game_data = None
    if args.game_data:
        game_data_path = Path(args.game_data)
        if game_data_path.exists():
            with open(game_data_path) as f:
                game_data = json.load(f)
            print(f"  Game data: {game_data_path}")

    # Infer game_id from baseline directory
    game_id = baseline_dir.name

    # Load baseline player states
    print(f"\nLoading baseline player states...")
    baseline_states = load_player_states(str(baseline_dir), args.end_turn, args.verbose)
    if baseline_states is None:
        print("Error: Could not load baseline player states")
        return 1

    if args.verbose:
        print(f"  Found {len(baseline_states)} players in baseline")
        for pid, state in sorted(baseline_states.items()):
            print(f"    Player {pid}: gold={state.get('gold', 0)}, techs={state.get('techs', 0)}")

    # Load fork player states
    print(f"\nLoading fork player states...")
    fork_states = load_player_states(str(fork_dir), args.end_turn, args.verbose)
    if fork_states is None:
        print("Error: Could not load fork player states")
        return 1

    if args.verbose:
        print(f"  Found {len(fork_states)} players in fork")
        for pid, state in sorted(fork_states.items()):
            print(f"    Player {pid}: gold={state.get('gold', 0)}, techs={state.get('techs', 0)}")

    # Get civ name from game data or baseline states
    civ_data = {}
    if game_data:
        civ_data = game_data.get("civilizations", {}).get(str(cond_dict["player_id"]), {})
    civ_name = civ_data.get("name") or baseline_states.get(cond_dict["player_id"], {}).get("nation", f"Player {cond_dict['player_id']}")

    # Create condition object
    condition = create_condition(
        condition_type=cond_dict["type"],
        player_id=cond_dict["player_id"],
        value=cond_dict["value"],
        civ_name=civ_name,
    )

    # Generate conditional questions
    print(f"\nGenerating conditional questions...")
    generator = ConditionalQuestionGenerator()

    # Build minimal game_data if not provided
    if game_data is None:
        # Create game_data from baseline states with time_series so generator
        # can find civilizations at checkpoint_turn
        game_data = {
            "civilizations": {
                str(pid): {
                    "name": state.get("nation", f"Player {pid}"),
                    "player_id": pid,
                }
                for pid, state in baseline_states.items()
            },
            # Fake time_series so generator can find players at checkpoint
            "time_series": {
                "score": {
                    str(args.checkpoint_turn): {
                        str(pid): 0 for pid in baseline_states.keys()
                    }
                }
            }
        }

    # Use treasury_comparative template (works with gold from savegame)
    # score_rank_1 requires score field not available in savegame parser
    cond_bank = generator.generate_conditional_bank(
        game_id=game_id,
        game_data=game_data,
        checkpoint_turn=args.checkpoint_turn,
        end_turn=args.end_turn,
        conditions=[condition],
        target_templates=["treasury_comparative"],
    )

    print(f"  Generated {len(cond_bank.questions)} conditional questions")

    # Resolve questions
    print(f"\nResolving questions...")

    baseline_outcome = ForkOutcome(
        fork_name="savegame_baseline",
        success=True,
        final_turn=args.end_turn,
        player_states=baseline_states,
        error=None,
    )

    intervention_outcome = ForkOutcome(
        fork_name="fork_intervention",
        success=True,
        final_turn=args.end_turn,
        player_states=fork_states,
        error=None,
    )

    effects = []
    for question in cond_bank.questions:
        answer_control = resolve_target_question(question, baseline_states)
        answer_intervention = resolve_target_question(question, fork_states)

        conditional_effect = None
        if answer_control is not None and answer_intervention is not None:
            conditional_effect = 1.0 if answer_control != answer_intervention else 0.0
            effects.append(conditional_effect)

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
            print(f"  {question.conditional_id}: control={answer_control}, intervention={answer_intervention}, effect={conditional_effect}")

    # Summary
    print(f"\nResults summary:")
    print(f"  Total questions: {len(cond_bank.questions)}")
    print(f"  Resolved: {len(effects)}")
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
