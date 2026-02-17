#!/usr/bin/env python3
"""Re-score continuous conditional results with corrected ground truth.

Fixes a bug where 3 of 6 continuous conditional templates used wrong units
for ground truth extraction from savegames:
  - population: used Freeciv population metric (~15x too high)
  - territory: used landarea metric (~1000x too high)
  - scores: used proxy formula (~3.7x too high)

This script:
1. Loads existing result JSON files
2. For each affected question, re-extracts ground truth from the fork savegame
   using corrected fields (citizen_population, tile_count, total_score)
3. Recomputes CRPS and MAE per model
4. Writes corrected result files

Usage:
    uv run python scripts/rescore_continuous.py \
        --results-dir data/results/ \
        --recordings-dir logs/recordings/
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from civrealm.world_reports.utils.savegame_parser import (
    find_local_savegame_for_turn,
    load_local_savegame,
    decompress_savegame_content,
    parse_player_states_for_conditional,
)
from civrealm.metrics import compute_crps, compute_aggregate_crps, compute_mae, compute_aggregate_mae


# Templates affected by the bug
AFFECTED_TEMPLATES = {
    "conditional_population_continuous",
    "conditional_territory_continuous",
    "conditional_scores_continuous",
}

# Map from result file name pattern to fork directory pattern
CONDITION_FORK_PATTERNS = {
    "republic": "forkgovRepublicp0",
    "gold500": "forkgoldadd500p0",
}


def extract_resolution_turn(question_text: str) -> int | None:
    """Extract resolution turn from question text like '...at turn 90?'"""
    match = re.search(r'turn (\d+)', question_text)
    return int(match.group(1)) if match else None


def extract_player_name(question_text: str) -> str | None:
    """Extract player/civ name from question text."""
    # Patterns like "what will Dacian's population be" or "how many tiles will Dacian control"
    match = re.search(r"what will (\w+)'s|how many \w+ will (\w+)", question_text)
    if match:
        return match.group(1) or match.group(2)
    return None


def get_corrected_ground_truth(
    template_id: str,
    player_states: dict[int, dict],
    player_id: int,
) -> float | None:
    """Get corrected ground truth value from player states.

    Uses the new corrected fields:
    - citizen_population: sum of citizen types (game-state units)
    - tile_count: owned tiles from map (game-state units)
    - total_score: actual Freeciv score (game-state units)
    """
    state = player_states.get(player_id, {})
    if not state:
        return None

    if template_id == "conditional_population_continuous":
        return state.get("citizen_population")
    elif template_id == "conditional_territory_continuous":
        return state.get("tile_count")
    elif template_id == "conditional_scores_continuous":
        return state.get("total_score")
    return None


def load_fork_player_states(
    recordings_dir: Path,
    game_id: str,
    fork_suffix: str,
    turn: int,
) -> dict[int, dict] | None:
    """Load player states from a fork savegame."""
    fork_dir = recordings_dir / f"{game_id}{fork_suffix}"
    if not fork_dir.exists():
        return None

    username = fork_dir.name
    savegame_name = find_local_savegame_for_turn(username, turn, str(fork_dir))
    if not savegame_name:
        return None

    result = load_local_savegame(savegame_name, str(fork_dir))
    if not result:
        return None

    content = decompress_savegame_content(result[0], result[1])
    return parse_player_states_for_conditional(content)


def find_player_id_by_name(player_states: dict[int, dict], name: str) -> int | None:
    """Find player ID by name in player states."""
    for pid, state in player_states.items():
        if state.get("name", "").lower() == name.lower():
            return pid
        # Also check nation
        if state.get("nation", "").lower() == name.lower():
            return pid
    return None


def rescore_result_file(
    result_path: Path,
    recordings_dir: Path,
    fork_suffix: str,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict | None:
    """Rescore a single result file with corrected ground truth.

    Returns the corrected data dict, or None if no changes needed.
    """
    with open(result_path) as f:
        data = json.load(f)

    questions = data.get("questions", [])
    if not questions:
        print(f"  No questions found in {result_path.name}")
        return None

    # Track changes
    changes = 0
    # Cache: (game_id, turn) -> player_states
    states_cache: dict[tuple[str, int], dict] = {}

    for q in questions:
        template_id = q.get("template_id", "")
        if template_id not in AFFECTED_TEMPLATES:
            continue

        game_id = q["game_id"]
        resolution_turn = extract_resolution_turn(q["question_text"])
        if resolution_turn is None:
            if verbose:
                print(f"  Warning: Could not extract resolution turn from {q['question_id']}")
            continue

        # Load player states (cached)
        cache_key = (game_id, resolution_turn)
        if cache_key not in states_cache:
            states = load_fork_player_states(
                recordings_dir, game_id, fork_suffix, resolution_turn
            )
            states_cache[cache_key] = states

        player_states = states_cache[cache_key]
        if player_states is None:
            if verbose:
                print(f"  Warning: No savegame for {game_id} turn {resolution_turn}")
            continue

        # Find player ID from question text
        player_name = extract_player_name(q["question_text"])
        if player_name is None:
            if verbose:
                print(f"  Warning: Could not extract player name from {q['question_id']}")
            continue

        player_id = find_player_id_by_name(player_states, player_name)
        if player_id is None:
            if verbose:
                print(f"  Warning: Could not find player '{player_name}' in states for {game_id}")
            continue

        # Get corrected ground truth
        corrected_gt = get_corrected_ground_truth(template_id, player_states, player_id)
        if corrected_gt is None:
            if verbose:
                print(f"  Warning: Could not get corrected GT for {q['question_id']}")
            continue

        old_gt = q["ground_truth"]
        if old_gt != corrected_gt:
            if verbose:
                print(f"  {q['question_id']} ({game_id} T{resolution_turn}): "
                      f"{template_id.split('_')[1]} GT {old_gt} -> {corrected_gt}")
            q["ground_truth"] = corrected_gt
            changes += 1

    if changes == 0:
        print(f"  No changes needed for {result_path.name}")
        return None

    print(f"  Fixed {changes} ground truth values in {result_path.name}")

    # Recompute model_results (CRPS and MAE)
    model_results = {}
    models = data.get("metadata", {}).get("models", [])

    for model in models:
        all_percentiles = []
        all_p50 = []
        all_true_values = []
        crps_by_template: dict[str, list] = {}
        mae_by_template: dict[str, list] = {}

        for q in questions:
            template_id = q["template_id"]
            gt = q["ground_truth"]
            pred = q.get("predictions", {}).get(model)

            if pred is None or pred.get("error"):
                continue

            percentiles = pred.get("percentiles")
            if percentiles is None:
                continue

            # Collect for aggregate
            all_percentiles.append(percentiles)
            all_p50.append(percentiles["p50"])
            all_true_values.append(gt)

            # Collect by template
            if template_id not in crps_by_template:
                crps_by_template[template_id] = {"percentiles": [], "p50": [], "gt": []}
                mae_by_template[template_id] = {"percentiles": [], "p50": [], "gt": []}
            crps_by_template[template_id]["percentiles"].append(percentiles)
            crps_by_template[template_id]["p50"].append(percentiles["p50"])
            crps_by_template[template_id]["gt"].append(gt)

        # Compute aggregates
        crps = compute_aggregate_crps(all_percentiles, all_true_values) if all_percentiles else float('nan')
        mae = compute_aggregate_mae(all_p50, all_true_values) if all_p50 else float('nan')

        # Compute per-template
        crps_template = {}
        mae_template = {}
        for tid, vals in crps_by_template.items():
            crps_template[tid] = compute_aggregate_crps(vals["percentiles"], vals["gt"])
            mae_template[tid] = compute_aggregate_mae(vals["p50"], vals["gt"])

        model_results[model] = {
            "binary": {
                "brier_score": float('nan'),
                "brier_by_template": {},
                "ece": float('nan'),
                "num_predictions": 0,
                "num_failures": 0,
            },
            "continuous": {
                "crps": crps,
                "mae": mae,
                "crps_by_template": crps_template,
                "mae_by_template": mae_template,
                "num_predictions": len(all_percentiles),
                "num_failures": sum(
                    1 for q in questions
                    if q.get("predictions", {}).get(model, {}).get("error")
                ),
            },
        }

    data["model_results"] = model_results

    return data


def main():
    parser = argparse.ArgumentParser(description="Re-score continuous conditional results")
    parser.add_argument("--results-dir", required=True, help="Directory containing result JSON files")
    parser.add_argument("--recordings-dir", required=True, help="Directory containing recording directories")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed progress")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    recordings_dir = Path(args.recordings_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    if not recordings_dir.exists():
        print(f"Error: Recordings directory not found: {recordings_dir}")
        return 1

    # Find all continuous conditional result files
    result_files = sorted(results_dir.glob("*conditional_continuous*.json"))
    if not result_files:
        print(f"No continuous conditional result files found in {results_dir}")
        return 1

    print(f"Found {len(result_files)} result files to process")

    for result_path in result_files:
        print(f"\nProcessing {result_path.name}...")

        # Determine fork suffix from filename
        fork_suffix = None
        for pattern, suffix in CONDITION_FORK_PATTERNS.items():
            if pattern in result_path.stem:
                fork_suffix = suffix
                break

        if fork_suffix is None:
            print(f"  Warning: Could not determine condition type from {result_path.name}, skipping")
            continue

        corrected = rescore_result_file(
            result_path, recordings_dir, fork_suffix,
            dry_run=args.dry_run, verbose=args.verbose,
        )

        if corrected and not args.dry_run:
            # Write corrected file (overwrite original)
            with open(result_path, 'w') as f:
                json.dump(corrected, f, indent=2)
            print(f"  Written corrected results to {result_path}")

            # Print summary of new CRPS values
            print(f"  New model CRPS values:")
            for model, results in corrected["model_results"].items():
                crps = results["continuous"]["crps"]
                print(f"    {model}: CRPS={crps:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
