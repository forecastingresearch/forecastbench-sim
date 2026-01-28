#!/usr/bin/env python3
"""Generate a readable TXT world report plus territory maps.

This script builds a human-readable TXT report from game data (data/games/)
and recordings. It also renders territory maps every N turns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for image generation

from civrealm.world_reports.config import ReportConfig
from civrealm.world_reports.data_loader import DataLoader
from civrealm.world_reports.extractors.metrics_collector import MetricsCollector
from civrealm.world_reports.utils.visualizations import MapVisualizer

EVENT_TYPES = [
    "city_founded",
    "city_conquered",
    "city_destroyed",
    "government_change",
    "diplomatic_change",
    "tech_discovered",
    "wonder_completed",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _format_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    widths = []
    for idx, header in enumerate(headers):
        max_width = len(str(header))
        for row in rows:
            max_width = max(max_width, len(str(row[idx])))
        widths.append(max_width)

    lines = []
    lines.append(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    return lines


def _build_turn_list(turns: list[int], sample_interval: int) -> list[int]:
    if not turns:
        return []
    max_turn = max(turns)
    sampled = {t for t in turns if t % sample_interval == 0}
    sampled.add(1)
    sampled.add(max_turn)
    return sorted(sampled)


def _get_ts_value(metric_data: dict[str, Any], turn: int, civ_id: int) -> Any:
    turn_data = metric_data.get(str(turn), {})
    return turn_data.get(str(civ_id), turn_data.get(civ_id))


def _format_time_series(
    title: str,
    metric_data: dict[str, Any],
    civs: list[tuple[int, dict[str, Any]]],
    turns: list[int],
) -> list[str]:
    lines = [title, f"turns: {', '.join(str(t) for t in turns)}"]
    for civ_id, civ in civs:
        values = []
        for turn in turns:
            val = _get_ts_value(metric_data, turn, civ_id)
            values.append("NA" if val is None else str(val))
        lines.append(f"{civ_id} {civ.get('name', f'Player {civ_id}')}: {', '.join(values)}")
    return lines


def _derive_scores_from_states(states: dict[int, dict[str, Any]], civ_ids: list[int]) -> dict[int, dict[int, int]]:
    scores_by_turn: dict[int, dict[int, int]] = {}
    for turn, state in states.items():
        players = state.get("player", {})
        scores = {}
        for civ_id in civ_ids:
            player_info = players.get(str(civ_id), {})
            score = player_info.get("score")
            if score is not None:
                scores[civ_id] = int(score)
        if scores:
            scores_by_turn[turn] = scores
    return scores_by_turn


def _format_scores(scores_by_turn: dict[int, dict[int, int]], civs: list[tuple[int, dict[str, Any]]], turns: list[int]) -> list[str]:
    lines = ["SCORES (derived from recordings)", f"turns: {', '.join(str(t) for t in turns)}"]
    for civ_id, civ in civs:
        values = []
        for turn in turns:
            val = scores_by_turn.get(turn, {}).get(civ_id)
            values.append("NA" if val is None else str(val))
        lines.append(f"{civ_id} {civ.get('name', f'Player {civ_id}')}: {', '.join(values)}")
    return lines


def _format_rankings(scores_by_turn: dict[int, dict[int, int]], civs: list[tuple[int, dict[str, Any]]], turns: list[int]) -> list[str]:
    civ_names = {cid: civ.get("name", f"Player {cid}") for cid, civ in civs}
    lines = ["RANKINGS (derived from scores)"]
    for turn in turns:
        scores = scores_by_turn.get(turn)
        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_str = ", ".join(
            f"{i + 1}={civ_names.get(cid, cid)}({score})"
            for i, (cid, score) in enumerate(ranked)
        )
        lines.append(f"Turn {turn}: {ranked_str}")
    return lines


def _format_events(
    events: list[dict[str, Any]],
    civ_names: dict[int, str],
    max_turn: int | None = None,
) -> list[str]:
    # Filter events by turn if max_turn is specified
    if max_turn is not None:
        events = [e for e in events if e.get("turn", 0) <= max_turn]

    if not events:
        return ["EVENTS", "No events recorded."]
    lines = ["EVENTS (chronological)", "Turn | Type | Civ | Description | Metadata"]
    lines.append("-" * 80)
    for event in sorted(events, key=lambda e: e.get("turn", 0)):
        turn = event.get("turn", "")
        event_type = event.get("type", "")
        civ_id = event.get("player_id")
        civ_label = civ_names.get(int(civ_id), f"Player {civ_id}") if civ_id is not None else "-"
        desc = event.get("description", "")
        meta = event.get("metadata") or {}
        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else "-"
        lines.append(f"{turn} | {event_type} | {civ_label} | {desc} | {meta_str}")
    return lines


def _format_event_types() -> list[str]:
    lines = ["EVENT TYPES (exhaustive)"]
    lines.append(", ".join(EVENT_TYPES))
    lines.append("If a type does not appear in EVENTS, it did not occur in the analyzed turns.")
    return lines


def _format_government_timeline(
    gov_by_turn: dict[int, dict[int, str]],
    civs: list[tuple[int, dict[str, Any]]],
) -> list[str]:
    lines = [
        "GOVERNMENT TIMELINE (derived from recordings)",
        "Each entry is a change point (turn number -> new government); the first entry is the first observed government.",
    ]
    for civ_id, civ in civs:
        name = civ.get("name", f"Player {civ_id}")
        last_gov = None
        change_points = []
        for turn in sorted(gov_by_turn.keys()):
            gov = gov_by_turn[turn].get(civ_id)
            if gov is None:
                continue
            if gov != last_gov:
                change_points.append(f"{turn} {gov}")
                last_gov = gov
        if change_points:
            lines.append(f"{civ_id} {name}: " + "; ".join(change_points))
        else:
            lines.append(f"{civ_id} {name}: NA (no changes observed)")
    return lines


def _derive_government_from_states(states: dict[int, dict[str, Any]], civ_ids: list[int]) -> dict[int, dict[int, str]]:
    gov_by_turn: dict[int, dict[int, str]] = {}
    for turn, state in states.items():
        players = state.get("player", {})
        turn_gov = {}
        for civ_id in civ_ids:
            player_info = players.get(str(civ_id), {})
            gov = player_info.get("government_name")
            if gov:
                turn_gov[civ_id] = gov
        if turn_gov:
            gov_by_turn[turn] = turn_gov
    return gov_by_turn


# NOTE: Diplomacy formatting commented out due to asymmetric/incorrect data in game_data.json files.
# The savegame parser stores both directions (X_Y and Y_X) which can have inconsistent states.
# See verify_ground_truth.py output for details.
#
# def _format_diplomacy(diplomacy: dict[str, Any], civs: list[tuple[int, dict[str, Any]]]) -> list[str]:
#     civ_names = {cid: civ.get("name", f"Player {cid}") for cid, civ in civs}
#     relations = diplomacy.get("relations", {})
#     if not relations:
#         return ["DIPLOMACY", "No diplomacy data available (savegames missing or not parsed)."]
#
#     lines = ["DIPLOMACY (state change timeline)"]
#     for pair_key in sorted(relations.keys()):
#         turn_data = relations[pair_key]
#         if not turn_data:
#             continue
#         parts = pair_key.split("_")
#         if len(parts) != 2:
#             continue
#         p1, p2 = int(parts[0]), int(parts[1])
#         header = f"PAIR {p1}-{civ_names.get(p1, p1)} vs {p2}-{civ_names.get(p2, p2)}"
#         lines.append(header)
#
#         last_state = None
#         changes = []
#         for turn in sorted(turn_data.keys()):
#             data = turn_data[turn]
#             state = data.get("state", "Unknown")
#             if state != last_state:
#                 changes.append(f"{turn} {state}")
#                 last_state = state
#         if changes:
#             lines.append("  State: " + "; ".join(changes))
#         else:
#             lines.append("  State: NA")
#     return lines


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("\n".join(lines))


def generate_txt_report(
    game_data_path: Path,
    recording_dir: Path,
    output_dir: Path,
    turn: int,
    sample_interval: int,
    map_interval: int,
) -> tuple[Path, Path]:
    data = _load_json(game_data_path)
    available_turn = data.get("metadata", {}).get("turn", turn)
    report_turn = min(turn, available_turn)
    turns_analyzed = data.get("metadata", {}).get(
        "turns_analyzed", list(range(1, report_turn + 1))
    )
    turns_analyzed = [t for t in turns_analyzed if t <= report_turn]
    if not turns_analyzed:
        turns_analyzed = [report_turn]

    civs = [(int(cid), c) for cid, c in sorted(data.get("civilizations", {}).items(), key=lambda x: int(x[0]))]
    civ_names = {cid: civ.get("name", f"Player {cid}") for cid, civ in civs}

    data_loader = DataLoader(str(recording_dir))
    states = data_loader.get_states_range(1, report_turn)

    scores_by_turn = _derive_scores_from_states(states, [cid for cid, _ in civs])
    gov_by_turn = _derive_government_from_states(states, [cid for cid, _ in civs])

    config = ReportConfig(
        recording_dir=str(recording_dir),
        output_dir=str(output_dir),
        report_turns=[report_turn],
        formats=["html"],
    )
    collector = MetricsCollector()
    # NOTE: Diplomacy collection commented out - data has asymmetric state issues
    # diplomacy = collector.collect_diplomacy(states, config, {cid: civ for cid, civ in civs})

    sampled_turns = _build_turn_list(turns_analyzed, sample_interval)
    full_turns = sorted(turns_analyzed)

    # Map size from recording if missing in JSON metadata
    map_size = data.get("metadata", {}).get("map_size", [0, 0])
    if map_size == [0, 0]:
        state = states.get(report_turn)
        if state and "map" in state and "tile_owner" in state["map"]:
            tile_owner = state["map"]["tile_owner"]
            if tile_owner and isinstance(tile_owner, list) and isinstance(tile_owner[0], list):
                map_size = [len(tile_owner), len(tile_owner[0])]

    lines = []
    lines.append("WORLD REPORT TXT v1")
    lines.append(f"Game ID: {data.get('metadata', {}).get('username', 'unknown')}")
    lines.append(f"Snapshot turn: {report_turn}")
    lines.append(f"Turns analyzed: {min(turns_analyzed)}..{max(turns_analyzed)}")
    lines.append(f"Map size: {map_size[0]}x{map_size[1]}")
    lines.append("")

    # Civilization list
    lines.append("CIVILIZATIONS")
    civ_rows = [[cid, civ.get("name", f"Player {cid}"), civ.get("adjective", ""), civ.get("nation_id", "")]
                for cid, civ in civs]
    lines.extend(_format_table(["ID", "Name", "Adjective", "Nation ID"], civ_rows))
    lines.append("")

    # Current state table
    snapshot = data.get("snapshots", {}).get(str(report_turn), {})
    current_rows = []
    for cid, civ in civs:
        current_rows.append([
            cid,
            civ.get("name", f"Player {cid}"),
            scores_by_turn.get(report_turn, {}).get(cid, snapshot.get("scores", {}).get(str(cid), "NA")),
            next((r.get("rank") for r in snapshot.get("rankings", []) if r.get("player_id") == cid), "NA"),
            gov_by_turn.get(report_turn, {}).get(cid, "NA"),
            _get_ts_value(data.get("time_series", {}).get("treasury", {}), report_turn, cid),
            _get_ts_value(data.get("time_series", {}).get("population", {}), report_turn, cid),
            _get_ts_value(data.get("time_series", {}).get("techs_known", {}), report_turn, cid),
            _get_ts_value(data.get("time_series", {}).get("wonders_count", {}), report_turn, cid),
            _get_ts_value(data.get("time_series", {}).get("cities_count", {}), report_turn, cid),
            _get_ts_value(data.get("time_series", {}).get("territory_size", {}), report_turn, cid),
            _get_ts_value(data.get("time_series", {}).get("units_count", {}), report_turn, cid),
            _get_ts_value(data.get("time_series", {}).get("military_units_count", {}), report_turn, cid),
            _get_ts_value(data.get("time_series", {}).get("science", {}), report_turn, cid),
            _get_ts_value(data.get("time_series", {}).get("trade_production", {}), report_turn, cid),
            _get_ts_value(data.get("time_series", {}).get("food_production", {}), report_turn, cid),
            _get_ts_value(data.get("time_series", {}).get("shield_production", {}), report_turn, cid),
        ])

    lines.append(f"CURRENT STATE (TURN {report_turn})")
    lines.extend(_format_table(
        ["ID", "Name", "Score", "Rank", "Government", "Treasury", "Population", "Techs", "Wonders",
         "Cities", "Territory", "Units", "Military", "Science", "Trade", "Food", "Shields"],
        current_rows
    ))
    lines.append("")

    # Government timeline
    lines.extend(_format_government_timeline(gov_by_turn, civs))
    lines.append("")

    # NOTE: Diplomacy section commented out - data has asymmetric state issues
    # # Diplomacy
    # lines.extend(_format_diplomacy(diplomacy, civs))
    # lines.append("")

    # Time series (sampled)
    ts = data.get("time_series", {})
    lines.extend(_format_time_series(f"TREASURY (sampled every {sample_interval} turns)", ts.get("treasury", {}), civs, sampled_turns))
    lines.append("")
    for metric_name, label in [
        ("population", "POPULATION"),
        ("techs_known", "TECHS KNOWN"),
        ("wonders_count", "WONDERS DISCOVERED"),
        ("cities_count", "CITIES COUNT"),
        ("territory_size", "TERRITORY SIZE"),
        ("science", "SCIENCE"),
        ("trade_production", "TRADE PRODUCTION"),
        ("food_production", "FOOD PRODUCTION"),
        ("shield_production", "SHIELD PRODUCTION"),
        ("units_count", "UNITS COUNT"),
        ("military_units_count", "MILITARY UNITS COUNT"),
    ]:
        lines.extend(_format_time_series(f"{label} (sampled every {sample_interval} turns)", ts.get(metric_name, {}), civs, sampled_turns))
        lines.append("")

    # Scores + rankings
    lines.extend(_format_scores(scores_by_turn, civs, sampled_turns))
    lines.append("")
    lines.extend(_format_rankings(scores_by_turn, civs, sampled_turns))
    lines.append("")

    # Events (filtered to snapshot turn to prevent leakage)
    lines.extend(_format_event_types())
    lines.append("")
    lines.extend(_format_events(data.get("events", []), civ_names, max_turn=report_turn))
    lines.append("")

    # Territory maps
    map_dir = output_dir / f"turn_{report_turn:03d}_territory_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    visualizer = MapVisualizer(dpi=150, style="seaborn", data_loader=data_loader)
    map_turns = [t for t in range(map_interval, report_turn + 1, map_interval)]
    if report_turn not in map_turns:
        map_turns.append(report_turn)
    lines.append("TERRITORY MAPS (PNG)")
    if not map_turns:
        lines.append("No map turns available.")
    else:
        for t in sorted(set(map_turns)):
            state = data_loader.get_state(t)
            if not state or "map" not in state or "player" not in state:
                lines.append(f"Turn {t}: missing state data (map not generated)")
                continue
            img_buf = visualizer.render_territory_map(
                map_state=state["map"],
                player_state=state["player"],
                title=f"Territory Map - Turn {t}",
            )
            img_path = map_dir / f"territory_turn_{t:03d}.png"
            img_path.write_bytes(img_buf.getvalue())
            lines.append(f"Turn {t}: {img_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / f"turn_{report_turn:03d}_report.txt"
    _write_lines(txt_path, lines)
    return txt_path, map_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate TXT world report and territory maps.")
    parser.add_argument("--game-id", required=True, help="Game ID (e.g., s12202)")
    parser.add_argument("--turn", type=int, default=50, help="Snapshot turn (default: 50)")
    parser.add_argument("--sample-interval", type=int, default=5, help="Sampling interval for most time series")
    parser.add_argument("--map-interval", type=int, default=10, help="Map generation interval in turns")
    parser.add_argument("--games-dir", type=Path, default=Path("data/games"), help="Directory containing game data files")
    parser.add_argument("--recordings-dir", type=Path, default=Path("logs/recordings"), help="Recordings base directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for TXT and maps")
    args = parser.parse_args()

    game_id = args.game_id
    # Load from data/games/{game_id}_data.json (source of truth)
    game_data_path = args.games_dir / f"{game_id}_data.json"
    recording_dir = args.recordings_dir / game_id
    output_dir = args.output_dir or (Path("data/questions") / game_id / "world_report")

    if not game_data_path.exists():
        raise FileNotFoundError(f"Game data not found: {game_data_path}")
    if not recording_dir.exists():
        raise FileNotFoundError(f"Recording directory not found: {recording_dir}")

    txt_path, map_dir = generate_txt_report(
        game_data_path=game_data_path,
        recording_dir=recording_dir,
        output_dir=output_dir,
        turn=args.turn,
        sample_interval=args.sample_interval,
        map_interval=args.map_interval,
    )

    print(f"TXT report: {txt_path}")
    print(f"Territory maps: {map_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
