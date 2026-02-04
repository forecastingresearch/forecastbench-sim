#!/usr/bin/env python3
"""
Calibrate thresholds based on actual game data.

This script analyzes game data to compute percentiles for each signal,
helping you set thresholds that produce balanced True/False distributions.

Usage:
    python calibrate_thresholds.py reports/s42/turn_299_data.json reports/s41/turn_300_data.json
    python calibrate_thresholds.py reports/*/turn_*_data.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics

sys.path.insert(0, 'src')


def load_game_data(paths: list[str]) -> list[dict]:
    """Load game data from multiple files."""
    data = []
    for path in paths:
        try:
            with open(path) as f:
                data.append(json.load(f))
            print(f"Loaded: {path}")
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
    return data


def collect_signal_values(games: list[dict], signal_name: str) -> dict[int, list[float]]:
    """
    Collect values for a signal at each turn across all games and players.

    Returns: {turn: [values across all players and games]}

    Note: time_series structure is signal -> turn -> player_id -> value
    """
    values_by_turn = defaultdict(list)

    for game in games:
        time_series = game.get("time_series", {}).get(signal_name, {})

        # Check structure - could be turn->player or player->turn
        if time_series:
            first_key = list(time_series.keys())[0]
            first_value = time_series[first_key]

            if isinstance(first_value, dict):
                # Structure: turn -> player_id -> value
                for turn_str, player_values in time_series.items():
                    turn = int(turn_str)
                    for player_id, value in player_values.items():
                        if value is not None:
                            values_by_turn[turn].append(value)
            else:
                # Structure: player_id -> turn -> value (alternative)
                for player_id, player_data in time_series.items():
                    if isinstance(player_data, dict):
                        for turn_str, value in player_data.items():
                            turn = int(turn_str)
                            if value is not None:
                                values_by_turn[turn].append(value)

    return dict(values_by_turn)


def compute_percentiles(values: list[float], percentiles: list[int] = [10, 25, 50, 75, 90]) -> dict[int, float]:
    """Compute percentiles for a list of values."""
    if not values:
        return {}

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    result = {}
    for p in percentiles:
        idx = int(p / 100 * (n - 1))
        result[p] = sorted_vals[idx]

    return result


def analyze_signal(games: list[dict], signal_name: str, sample_turns: list[int]) -> dict:
    """Analyze a signal across games and compute statistics at sample turns."""
    values_by_turn = collect_signal_values(games, signal_name)

    results = {}
    for turn in sample_turns:
        # Get values at or near this turn
        nearby_values = []
        for t in range(max(1, turn - 5), turn + 6):
            if t in values_by_turn:
                nearby_values.extend(values_by_turn[t])

        if nearby_values:
            results[turn] = {
                "count": len(nearby_values),
                "min": min(nearby_values),
                "max": max(nearby_values),
                "mean": statistics.mean(nearby_values),
                "median": statistics.median(nearby_values),
                "percentiles": compute_percentiles(nearby_values),
            }

    return results


def suggest_thresholds(stats: dict, target_true_rate: float = 0.4) -> list[float]:
    """
    Suggest threshold values that would achieve approximately the target True rate.

    For "greater than" questions, we want thresholds where ~40% of values exceed them.
    This means using the (1 - target_true_rate) percentile, e.g., 60th percentile for 40% True.
    """
    percentile_for_target = int((1 - target_true_rate) * 100)

    thresholds = set()
    for turn, turn_stats in stats.items():
        percs = turn_stats.get("percentiles", {})
        # Use percentiles that would give reasonable True rates
        for p in [25, 50, 75]:
            if p in percs:
                thresholds.add(round(percs[p]))

    return sorted(thresholds)


def main():
    parser = argparse.ArgumentParser(description='Calibrate thresholds from game data')
    parser.add_argument('data_files', nargs='+', help='Game data JSON files')
    parser.add_argument('--target-true-rate', type=float, default=0.4,
                        help='Target True answer rate (default: 0.4)')
    parser.add_argument('--sample-turns', nargs='+', type=int,
                        default=[50, 100, 150, 200],
                        help='Turns to sample (default: 50 100 150 200)')
    args = parser.parse_args()

    # Expand glob patterns
    from glob import glob
    files = []
    for pattern in args.data_files:
        files.extend(glob(pattern))

    if not files:
        print("No files found!")
        sys.exit(1)

    # Load data
    print(f"\nLoading {len(files)} game data files...")
    games = load_game_data(files)

    if not games:
        print("No valid game data loaded!")
        sys.exit(1)

    print(f"\nAnalyzing {len(games)} games...")

    # Signals to analyze
    signals = [
        "techs_known",
        "population",
        "territory_size",
        "treasury",
        "cities_count",
        "military_units_count",
        "units_count",
    ]

    print("\n" + "=" * 70)
    print("SIGNAL ANALYSIS")
    print("=" * 70)

    suggested_thresholds = {}

    for signal in signals:
        print(f"\n--- {signal} ---")
        stats = analyze_signal(games, signal, args.sample_turns)

        if not stats:
            print("  No data available")
            continue

        for turn in sorted(stats.keys()):
            s = stats[turn]
            print(f"  Turn {turn}: n={s['count']}, "
                  f"range=[{s['min']:.0f}, {s['max']:.0f}], "
                  f"median={s['median']:.0f}, mean={s['mean']:.1f}")
            percs = s['percentiles']
            print(f"           percentiles: " +
                  ", ".join(f"p{p}={v:.0f}" for p, v in sorted(percs.items())))

        # Suggest thresholds
        suggested = suggest_thresholds(stats, args.target_true_rate)
        suggested_thresholds[signal] = suggested
        print(f"  Suggested thresholds: {suggested}")

    # Print configuration
    print("\n" + "=" * 70)
    print("SUGGESTED THRESHOLD CONFIGURATION")
    print("=" * 70)
    print("\nCopy this to thresholds.py:\n")
    print("DEFAULT_THRESHOLDS = ThresholdConfig(")
    print("    defaults={")
    for signal, thresholds in suggested_thresholds.items():
        if thresholds:
            print(f'        "{signal}": {thresholds},')
    print("    }")
    print(")")

    # Also compute event-based statistics
    print("\n" + "=" * 70)
    print("EVENT STATISTICS")
    print("=" * 70)

    event_counts = defaultdict(int)
    for game in games:
        for event in game.get("events", []):
            event_counts[event.get("type")] += 1

    total_turns = sum(g.get("metadata", {}).get("turn", 0) for g in games)
    total_civs = sum(g.get("metadata", {}).get("num_civilizations", 0) for g in games)

    print(f"\nAcross {len(games)} games, {total_turns} total turns, {total_civs} total civs:")
    for event_type, count in sorted(event_counts.items()):
        rate_per_turn = count / total_turns if total_turns > 0 else 0
        print(f"  {event_type}: {count} events ({rate_per_turn:.2f} per turn)")


if __name__ == '__main__':
    main()
