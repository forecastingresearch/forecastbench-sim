#!/usr/bin/env python3
"""
Compute statistical properties for all signals across game recordings.

This script extracts percentiles, growth rates, and event probabilities
that can be used to derive principled thresholds for question generation.

Usage:
    python compute_signal_statistics.py --data-dir data/games --output signal_stats.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm


# Key turns to compute statistics at (covering H1, H2, H3 resolution windows)
KEY_TURNS = [20, 30, 50, 70, 100, 125, 150, 175, 200]

# Percentiles to compute
PERCENTILES = [10, 25, 50, 75, 90]

# Time series signals to analyze
TIME_SERIES_SIGNALS = [
    "techs_known",
    "population",
    "territory_size",
    "treasury",
    "cities_count",
    "military_units_count",
    "wonders_count",
    "units_count",
]


def load_game_data(file_path: Path) -> dict | None:
    """Load and validate game data from JSON file."""
    try:
        with open(file_path) as f:
            data = json.load(f)
        if "time_series" not in data or "metadata" not in data:
            return None
        return data
    except (json.JSONDecodeError, IOError):
        return None


def get_signal_value(time_series: dict, turn: int, player_id: int) -> float | None:
    """Extract signal value for a player at a specific turn."""
    turn_data = time_series.get(str(turn), {})
    value = turn_data.get(str(player_id))
    if value is not None:
        return float(value)
    return None


def compute_time_series_stats(games: list[dict]) -> dict:
    """
    Compute percentile statistics for time series signals.

    Returns:
        Dictionary mapping signal -> turn -> percentile stats
    """
    # Collect values: signal -> turn -> list of values (across all games/players)
    signal_values = defaultdict(lambda: defaultdict(list))

    for game in games:
        time_series = game.get("time_series", {})
        max_turn = game["metadata"]["turn"]
        num_players = game["metadata"].get("num_civilizations", 9)

        for signal_name in TIME_SERIES_SIGNALS:
            if signal_name not in time_series:
                continue

            for turn in KEY_TURNS:
                if turn > max_turn:
                    continue

                # Collect values for all players at this turn
                for player_id in range(num_players):
                    value = get_signal_value(time_series[signal_name], turn, player_id)
                    if value is not None:
                        signal_values[signal_name][turn].append(value)

    # Compute percentiles
    stats = {}
    for signal_name, turn_values in signal_values.items():
        stats[signal_name] = {
            "by_turn": {},
            "description": f"Percentiles for {signal_name} at key turns"
        }

        for turn, values in sorted(turn_values.items()):
            if len(values) < 5:  # Need minimum samples
                continue

            arr = np.array(values)
            stats[signal_name]["by_turn"][turn] = {
                f"p{p}": float(np.percentile(arr, p)) for p in PERCENTILES
            }
            stats[signal_name]["by_turn"][turn]["mean"] = float(np.mean(arr))
            stats[signal_name]["by_turn"][turn]["std"] = float(np.std(arr))
            stats[signal_name]["by_turn"][turn]["n"] = len(values)

    return stats


def compute_growth_stats(games: list[dict]) -> dict:
    """
    Compute growth/change statistics for signals over different windows.

    This is useful for questions like "Will territory increase by X?"
    """
    # Windows to analyze (start_turn, end_turn)
    windows = [
        (50, 70),   # H1: 20 turns
        (50, 125),  # H2: 75 turns
        (50, 200),  # H3: 150 turns
    ]

    growth_values = defaultdict(lambda: defaultdict(list))

    for game in games:
        time_series = game.get("time_series", {})
        max_turn = game["metadata"]["turn"]
        num_players = game["metadata"].get("num_civilizations", 9)

        for signal_name in ["territory_size", "population", "techs_known", "cities_count"]:
            if signal_name not in time_series:
                continue

            for start, end in windows:
                if end > max_turn:
                    continue

                window_key = f"{start}_to_{end}"

                for player_id in range(num_players):
                    start_val = get_signal_value(time_series[signal_name], start, player_id)
                    end_val = get_signal_value(time_series[signal_name], end, player_id)

                    if start_val is not None and end_val is not None:
                        # Absolute change
                        change = end_val - start_val
                        growth_values[signal_name][window_key].append(change)

    # Compute percentiles for growth
    stats = {}
    for signal_name, window_values in growth_values.items():
        stats[signal_name] = {"windows": {}}

        for window_key, values in window_values.items():
            if len(values) < 5:
                continue

            arr = np.array(values)
            stats[signal_name]["windows"][window_key] = {
                f"p{p}": float(np.percentile(arr, p)) for p in PERCENTILES
            }
            stats[signal_name]["windows"][window_key]["mean"] = float(np.mean(arr))
            stats[signal_name]["windows"][window_key]["std"] = float(np.std(arr))
            stats[signal_name]["windows"][window_key]["n"] = len(values)

    return stats


def compute_event_stats(games: list[dict]) -> dict:
    """
    Compute event occurrence statistics.

    For each event type, compute:
    - Probability of occurrence within different time windows
    - Average frequency per turn
    """
    # Track events per game/player
    event_counts = defaultdict(lambda: defaultdict(list))

    # Track event occurrences in windows (from turn 50)
    window_occurrences = defaultdict(lambda: defaultdict(lambda: {"occurred": 0, "total": 0}))

    windows = {
        "H1": (50, 70),
        "H2": (50, 125),
        "H3": (50, 200),
    }

    for game in games:
        events = game.get("events", [])
        max_turn = game["metadata"]["turn"]
        num_players = game["metadata"].get("num_civilizations", 9)

        # Count events per player in each window
        player_events = defaultdict(lambda: defaultdict(set))  # player -> event_type -> set of turns

        for e in events:
            event_type = e.get("type")
            turn = e.get("turn", 0)
            player_id = e.get("player_id")

            if player_id is not None:
                try:
                    player_id = int(player_id)
                    player_events[player_id][event_type].add(turn)
                except (ValueError, TypeError):
                    pass

        # Check window occurrences for each player
        for player_id in range(num_players):
            for window_name, (start, end) in windows.items():
                if end > max_turn:
                    continue

                for event_type in ["city_founded", "city_conquered", "city_destroyed",
                                   "government_change", "tech_discovered"]:
                    window_occurrences[event_type][window_name]["total"] += 1

                    # Check if any event of this type occurred in window
                    event_turns = player_events[player_id].get(event_type, set())
                    if any(start < t <= end for t in event_turns):
                        window_occurrences[event_type][window_name]["occurred"] += 1

        # Also track wonder completions (global, not per-player)
        for window_name, (start, end) in windows.items():
            if end > max_turn:
                continue

            window_occurrences["wonder_completed_any"][window_name]["total"] += 1

            wonder_turns = [e.get("turn", 0) for e in events if e.get("type") == "wonder_completed"]
            if any(start < t <= end for t in wonder_turns):
                window_occurrences["wonder_completed_any"][window_name]["occurred"] += 1

    # Convert to probabilities
    stats = {}
    for event_type, windows_data in window_occurrences.items():
        stats[event_type] = {"probability_in_window": {}}

        for window_name, counts in windows_data.items():
            if counts["total"] > 0:
                prob = counts["occurred"] / counts["total"]
                stats[event_type]["probability_in_window"][window_name] = {
                    "probability": round(prob, 4),
                    "occurred": counts["occurred"],
                    "total": counts["total"],
                }

    return stats


def compute_war_stats(games: list[dict]) -> dict:
    """
    Compute war-related statistics from diplomacy data.
    """
    # Track war states at key turns
    war_any_at_turn = defaultdict(lambda: {"at_war": 0, "total": 0})
    war_dyad_at_turn = defaultdict(lambda: {"at_war": 0, "total": 0})

    for game in games:
        diplomacy = game.get("diplomacy", {})
        max_turn = game["metadata"]["turn"]
        num_players = game["metadata"].get("num_civilizations", 9)

        for turn in KEY_TURNS:
            if turn > max_turn:
                continue

            turn_diplomacy = diplomacy.get(str(turn), {})

            for player_id in range(num_players):
                player_diplo = turn_diplomacy.get(str(player_id), {})

                # Check if at war with any
                war_any_at_turn[turn]["total"] += 1
                at_war = False

                for other_id, status in player_diplo.items():
                    if status == "war":
                        at_war = True
                        break

                if at_war:
                    war_any_at_turn[turn]["at_war"] += 1

                # Count dyadic relationships
                for other_id in range(player_id + 1, num_players):
                    status = player_diplo.get(str(other_id), "unknown")
                    war_dyad_at_turn[turn]["total"] += 1
                    if status == "war":
                        war_dyad_at_turn[turn]["at_war"] += 1

    stats = {
        "at_war_any": {
            "by_turn": {
                turn: {
                    "probability": round(data["at_war"] / data["total"], 4) if data["total"] > 0 else 0,
                    "at_war": data["at_war"],
                    "total": data["total"],
                }
                for turn, data in sorted(war_any_at_turn.items())
            }
        },
        "at_war_dyad": {
            "by_turn": {
                turn: {
                    "probability": round(data["at_war"] / data["total"], 4) if data["total"] > 0 else 0,
                    "at_war": data["at_war"],
                    "total": data["total"],
                }
                for turn, data in sorted(war_dyad_at_turn.items())
            }
        },
    }

    return stats


def compute_score_rank_stats(games: list[dict]) -> dict:
    """
    Compute score ranking statistics.

    Track how often each player achieves rank 1 (highest score).
    """
    rank_1_at_turn = defaultdict(lambda: {"achieved": 0, "total": 0})

    for game in games:
        time_series = game.get("time_series", {})
        max_turn = game["metadata"]["turn"]
        num_players = game["metadata"].get("num_civilizations", 9)

        # Use a proxy for score - could be population, territory, or wonders
        # Let's use a composite or just population as proxy
        score_series = time_series.get("population", {})

        for turn in KEY_TURNS:
            if turn > max_turn:
                continue

            # Get all player scores at this turn
            scores = {}
            for player_id in range(num_players):
                val = get_signal_value(score_series, turn, player_id)
                if val is not None:
                    scores[player_id] = val

            if not scores:
                continue

            # Find rank 1 (highest score)
            max_score = max(scores.values())

            for player_id in range(num_players):
                rank_1_at_turn[turn]["total"] += 1
                if scores.get(player_id, 0) == max_score:
                    rank_1_at_turn[turn]["achieved"] += 1

    stats = {
        "score_rank_1": {
            "by_turn": {
                turn: {
                    "probability": round(data["achieved"] / data["total"], 4) if data["total"] > 0 else 0,
                    "achieved": data["achieved"],
                    "total": data["total"],
                }
                for turn, data in sorted(rank_1_at_turn.items())
            }
        }
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute signal statistics from game data")
    parser.add_argument("--data-dir", type=str, default="data/games",
                        help="Directory containing game JSON files")
    parser.add_argument("--output", type=str, default="signal_stats.json",
                        help="Output file for statistics")
    args = parser.parse_args()

    # Find all data files
    data_dir = Path(args.data_dir)
    data_files = sorted(data_dir.glob("*_data.json"))
    print(f"Found {len(data_files)} data files in {data_dir}")

    # Load all games
    games = []
    for f in tqdm(data_files, desc="Loading games"):
        data = load_game_data(f)
        if data:
            games.append(data)

    print(f"Loaded {len(games)} valid games")

    # Compute statistics
    print("\nComputing time series statistics...")
    time_series_stats = compute_time_series_stats(games)

    print("Computing growth statistics...")
    growth_stats = compute_growth_stats(games)

    print("Computing event statistics...")
    event_stats = compute_event_stats(games)

    print("Computing war statistics...")
    war_stats = compute_war_stats(games)

    print("Computing score rank statistics...")
    score_stats = compute_score_rank_stats(games)

    # Combine all stats
    all_stats = {
        "metadata": {
            "num_games": len(games),
            "key_turns": KEY_TURNS,
            "percentiles": PERCENTILES,
        },
        "time_series": time_series_stats,
        "growth": growth_stats,
        "events": event_stats,
        "war": war_stats,
        "rankings": score_stats,
    }

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: KEY STATISTICS FOR THRESHOLD CALIBRATION")
    print("=" * 70)

    print("\n--- TECH COUNT BY TURN (for tech_count_gte questions) ---")
    if "techs_known" in time_series_stats:
        for turn, stats in sorted(time_series_stats["techs_known"]["by_turn"].items()):
            print(f"  Turn {turn:3d}: p25={stats['p25']:5.1f}, p50={stats['p50']:5.1f}, p75={stats['p75']:5.1f} (n={stats['n']})")

    print("\n--- POPULATION BY TURN (for population_gte questions) ---")
    if "population" in time_series_stats:
        for turn, stats in sorted(time_series_stats["population"]["by_turn"].items()):
            print(f"  Turn {turn:3d}: p25={stats['p25']:5.1f}, p50={stats['p50']:5.1f}, p75={stats['p75']:5.1f} (n={stats['n']})")

    print("\n--- TERRITORY SIZE BY TURN (for territory_gte questions) ---")
    if "territory_size" in time_series_stats:
        for turn, stats in sorted(time_series_stats["territory_size"]["by_turn"].items()):
            print(f"  Turn {turn:3d}: p25={stats['p25']:5.1f}, p50={stats['p50']:5.1f}, p75={stats['p75']:5.1f} (n={stats['n']})")

    print("\n--- TERRITORY GROWTH (for territory_gain questions) ---")
    if "territory_size" in growth_stats:
        for window, stats in growth_stats["territory_size"]["windows"].items():
            print(f"  {window}: p25={stats['p25']:5.1f}, p50={stats['p50']:5.1f}, p75={stats['p75']:5.1f}")

    print("\n--- EVENT PROBABILITIES (for event questions) ---")
    for event_type, stats in event_stats.items():
        print(f"  {event_type}:")
        for window, data in stats.get("probability_in_window", {}).items():
            print(f"    {window}: {data['probability']*100:.1f}% ({data['occurred']}/{data['total']})")

    print("\n--- WAR PROBABILITIES ---")
    if "at_war_any" in war_stats:
        print("  at_war_any by turn:")
        for turn, data in war_stats["at_war_any"]["by_turn"].items():
            print(f"    Turn {turn}: {data['probability']*100:.1f}%")


if __name__ == "__main__":
    main()
