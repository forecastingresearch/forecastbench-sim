"""
Compress CivBench world report JSON into a token-efficient format for model input.

Takes the full ~260KB JSON and produces a ~15-30KB version suitable for LLM context windows.
Uses sampled time series (every N turns) to preserve trends while reducing size.
"""

from __future__ import annotations

import json
from typing import Any


def compress_time_series_sampled(
    time_series: dict,
    civ_ids: list[str],
    metrics: list[str] | None = None,
    sample_turns: list[int] | None = None,
) -> dict:
    """
    Compress time series by keeping only specific sample turns.

    Args:
        time_series: Full time series dict {metric: {turn: {civ_id: value}}}
        civ_ids: List of civilization IDs
        metrics: Which metrics to include (None = all)
        sample_turns: Which turns to keep (e.g., [1, 10, 20, 30, 40, 50])

    Returns:
        Compressed dict with sampled values per metric/civ
    """
    if metrics is None:
        metrics = list(time_series.keys())

    if not time_series:
        return {}

    # Determine max turn from data
    first_metric = next(iter(time_series.values()), {})
    max_turn = max((int(t) for t in first_metric.keys()), default=50)

    if sample_turns is None:
        # Default: every 10 turns plus first and last
        sample_turns = [1] + list(range(10, max_turn, 10)) + [max_turn]
        sample_turns = sorted(set(sample_turns))

    compressed = {"_turns": sample_turns}

    for metric in metrics:
        if metric not in time_series:
            continue

        metric_data = time_series[metric]
        compressed[metric] = {}

        for civ_id in civ_ids:
            values = []
            for turn in sample_turns:
                turn_str = str(turn)
                if turn_str in metric_data:
                    val = metric_data[turn_str].get(
                        civ_id, metric_data[turn_str].get(int(civ_id), 0)
                    )
                    values.append(val)
                else:
                    values.append(None)

            compressed[metric][civ_id] = values

    return compressed


def extract_current_state(data: dict, turn: int) -> dict:
    """Extract current state snapshot at given turn."""
    turn_str = str(turn)
    civ_ids = list(data.get("civilizations", {}).keys())

    state = {}

    # From snapshots if available
    if turn_str in data.get("snapshots", {}):
        snapshot = data["snapshots"][turn_str]
        for key in ["scores", "rankings"]:
            if key in snapshot:
                state[key] = snapshot[key]
        if "world_totals" in snapshot:
            state["world_totals"] = snapshot["world_totals"]

    # From time series - get final values
    key_metrics = [
        "treasury",
        "population",
        "techs_known",
        "territory_size",
        "cities_count",
        "units_count",
        "military_units_count",
        "food_production",
        "shield_production",
        "trade_production",
        "science",
    ]

    for metric in key_metrics:
        if metric in data.get("time_series", {}):
            metric_data = data["time_series"][metric]
            if turn_str in metric_data:
                # Convert to list ordered by civ_id
                values = []
                for civ_id in civ_ids:
                    val = metric_data[turn_str].get(
                        civ_id, metric_data[turn_str].get(int(civ_id), 0)
                    )
                    values.append(val)
                state[metric] = values

    return state


def compress_events(events: list[dict]) -> list[dict]:
    """Compress events to minimal representation."""
    compressed = []
    for event in events:
        e: dict[str, Any] = {
            "t": event["turn"],
            "type": event["type"][:4],  # Abbreviate: tech, city, wond, govt
            "civ": event.get("player_id"),
        }

        # Add type-specific minimal data
        if event["type"] == "tech_discovered":
            e["tech"] = event.get("metadata", {}).get("tech_name", "")
        elif event["type"] == "city_founded":
            e["city"] = event.get("metadata", {}).get("city_name", "")
            if "location" in event:
                e["loc"] = event["location"]
        elif event["type"] == "wonder_completed":
            e["wonder"] = event.get("metadata", {}).get("wonder_name", "")
        elif event["type"] == "government_change":
            e["from"] = event.get("metadata", {}).get("from", "")
            e["to"] = event.get("metadata", {}).get("to", "")

        compressed.append(e)

    return compressed


def compress_diplomacy(diplomacy: dict, current_turn: int) -> dict:
    """Compress diplomacy to current state + state changes only."""
    relations = diplomacy.get("relations", {})
    turn_str = str(current_turn)

    compressed: dict[str, Any] = {
        "current_state": {},
        "current_love": {},
        "state_changes": [],
    }

    for pair_id, pair_history in relations.items():
        # Get current state
        if turn_str in pair_history:
            current = pair_history[turn_str]
            compressed["current_state"][pair_id] = current["state"]
            compressed["current_love"][pair_id] = current["love"]

        # Find state changes
        prev_state = None
        turns = sorted(pair_history.keys(), key=int)
        for turn in turns:
            state = pair_history[turn]["state"]
            if state != prev_state:
                if prev_state is not None:  # Don't record initial state as a "change"
                    compressed["state_changes"].append(
                        {"t": int(turn), "pair": pair_id, "from": prev_state, "to": state}
                    )
                prev_state = state

    # Include thresholds for interpretation
    if "attitude_thresholds" in diplomacy:
        compressed["thresholds"] = diplomacy["attitude_thresholds"]

    return compressed


def compress_world_report(
    data: dict,
    sample_interval: int = 10,
) -> dict:
    """
    Compress a full world report JSON to token-efficient format.

    Uses "standard" compression mode:
    - Current state snapshot
    - Sampled time series history (every N turns)
    - Compressed events
    - Diplomacy with state changes

    Args:
        data: Full world report data
        sample_interval: Turns between samples (default: 10)

    Returns:
        Compressed world report dict (~10-15% of original size)
    """
    current_turn = data.get("metadata", {}).get("turn", 50)
    civ_ids = list(data.get("civilizations", {}).keys())

    # Build civilization name mapping
    civ_names = [data["civilizations"][cid]["name"] for cid in civ_ids]

    compressed: dict[str, Any] = {
        "metadata": {
            "turn": current_turn,
            "civs": civ_names,
        }
    }

    # Current state (always included)
    compressed["current"] = extract_current_state(data, current_turn)

    # Events (compressed)
    compressed["events"] = compress_events(data.get("events", []))

    # Sampled history
    key_metrics = [
        "treasury",
        "population",
        "techs_known",
        "territory_size",
        "cities_count",
        "military_units_count",
    ]

    max_turn = current_turn
    sample_turns = [1] + list(range(sample_interval, max_turn, sample_interval))
    if max_turn not in sample_turns:
        sample_turns.append(max_turn)

    compressed["history"] = compress_time_series_sampled(
        data.get("time_series", {}), civ_ids, key_metrics, sample_turns
    )

    # Diplomacy
    if "diplomacy" in data:
        compressed["diplomacy"] = compress_diplomacy(data["diplomacy"], current_turn)

    return compressed


def load_and_compress_world_report(report_path: str) -> str:
    """
    Load a world report JSON file and return compressed JSON string.

    Args:
        report_path: Path to the world report JSON file

    Returns:
        Compact JSON string of compressed world report
    """
    with open(report_path) as f:
        data = json.load(f)

    compressed = compress_world_report(data)
    return json.dumps(compressed, separators=(",", ":"))
