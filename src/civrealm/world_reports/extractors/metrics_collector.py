"""Metrics collection from game states and savegames

This module extracts all world report data from state files and savegames,
organizing it into a flat, metric-based structure.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from ..utils import metrics
from ..utils.event_detector import EventDetector
from ..utils.savegame_parser import get_savegame_data_for_report


def select_snapshot_turns(max_turn: int, max_snapshots: int = 5,
                         min_turn: int = 5, min_spacing: int = 10) -> List[int]:
    """Select snapshot turns for territorial maps using backward selection

    Args:
        max_turn: The latest turn in the report
        max_snapshots: Maximum number of snapshots to include (default: 5)
        min_turn: Earliest turn to consider (default: 5)
        min_spacing: Minimum turns between snapshots (default: 10)

    Returns:
        Sorted list of turn numbers for snapshot generation
    """
    if max_turn < min_turn:
        return []

    if max_turn == min_turn:
        return [min_turn]

    available_range = max_turn - min_turn
    ideal_spacing = available_range // (max_snapshots - 1) if max_snapshots > 1 else available_range
    spacing = max(min_spacing, ideal_spacing)

    snapshots = [max_turn]
    current = max_turn

    while len(snapshots) < max_snapshots:
        next_turn = current - spacing
        if next_turn < min_turn:
            if min_turn not in snapshots and (current - min_turn) >= min_spacing:
                snapshots.append(min_turn)
            break
        snapshots.append(next_turn)
        current = next_turn

    return sorted(snapshots)


class MetricsCollector:
    """Collects and organizes all world report metrics from game data"""

    def __init__(self):
        """Initialize metrics collector"""
        pass

    def collect_all(
        self,
        states: Dict[int, Dict],
        config: Any,
        data_loader: Any
    ) -> Dict[str, Any]:
        """Collect all metrics and return complete data structure

        Args:
            states: Dict mapping turn numbers to game states
            config: ReportConfig instance
            data_loader: DataLoader instance

        Returns:
            Complete world report data dictionary
        """
        if not states:
            raise ValueError("No states provided")

        sorted_turns = sorted(states.keys())
        max_turn = max(sorted_turns)

        print("  Collecting metadata...")
        metadata = self.collect_metadata(states, config)

        print("  Collecting civilizations...")
        civilizations = self.collect_civilizations(states, data_loader, config)

        # Update civilization count in metadata to match what we're actually displaying
        metadata["num_civilizations"] = len(civilizations)

        print("  Collecting time series data...")
        time_series = self.collect_time_series(states, config, civilizations)

        print("  Detecting events...")
        events = self.collect_events(states, data_loader, config)

        print("  Collecting snapshots...")
        snapshots = self.collect_snapshots(states, max_turn)

        print("  Determining territory snapshot turns...")
        territory_snapshots = {
            "turns": select_snapshot_turns(max_turn),
            "note": "Territory maps generated during rendering from state files"
        }

        return {
            "metadata": metadata,
            "civilizations": civilizations,
            "time_series": time_series,
            "events": events,
            "snapshots": snapshots,
            "territory_snapshots": territory_snapshots
        }

    def collect_metadata(self, states: Dict[int, Dict], config: Any) -> Dict[str, Any]:
        """Collect report metadata

        Args:
            states: Dict mapping turn numbers to game states
            config: ReportConfig instance

        Returns:
            Metadata dictionary
        """
        sorted_turns = sorted(states.keys())
        max_turn = max(sorted_turns)
        last_state = states[max_turn]

        # Extract username from recording_dir if available
        username = "unknown"
        if hasattr(config, 'recording_dir'):
            # Recording dir is typically: logs/recordings/username/
            parts = config.recording_dir.rstrip('/').split('/')
            if len(parts) >= 2:
                username = parts[-1]

        # Map size
        map_state = last_state.get('map', {})
        map_size = [
            map_state.get('xsize', 0),
            map_state.get('ysize', 0)
        ]

        # Count civilizations
        players = last_state.get('player', {})
        num_civs = len([p for p in players.values() if isinstance(p, dict)])

        return {
            "turn": max_turn,
            "turns_analyzed": sorted_turns,
            "generated_at": datetime.now().isoformat(),
            "username": username,
            "map_size": map_size,
            "num_civilizations": num_civs
        }

    def collect_civilizations(
        self,
        states: Dict[int, Dict],
        data_loader: Any,
        config: Any
    ) -> Dict[int, Dict[str, Any]]:
        """Collect civilization information

        Args:
            states: Dict mapping turn numbers to game states
            data_loader: DataLoader instance
            config: ReportConfig instance

        Returns:
            Dict mapping player_id to civilization info
        """
        civilizations = {}

        # Collect all player IDs
        all_player_ids = set()
        for state in states.values():
            if 'player' in state:
                for pid in state['player'].keys():
                    all_player_ids.add(int(pid))

        # Get names and nation info for each player
        for player_id in all_player_ids:
            # Find the latest state where this player exists
            for turn in sorted(states.keys(), reverse=True):
                state = states[turn]
                if 'player' not in state:
                    continue

                pid_str = str(player_id)
                player_info = state['player'].get(pid_str) or state['player'].get(player_id)

                if isinstance(player_info, dict):
                    nation_id = player_info.get('nation')
                    civ_name = self._get_nation_name(nation_id, data_loader)

                    # If no civilization name, try savegame data
                    if not civ_name:
                        sorted_turns = sorted(states.keys())
                        for turn in reversed(sorted_turns):
                            savegame_data = get_savegame_data_for_report(config, turn)
                            if savegame_data and 'nations' in savegame_data:
                                if player_id in savegame_data['nations']:
                                    nation_identifier = savegame_data['nations'][player_id]
                                    civ_name = self._get_nation_name_from_savegame(
                                        nation_identifier, data_loader
                                    )
                                    if civ_name:
                                        break

                    # Fallback to player name
                    if not civ_name:
                        civ_name = player_info.get('name', f'Player {player_id}')

                    civilizations[player_id] = {
                        "name": civ_name,
                        "adjective": civ_name,  # Same as name for now
                        "nation_id": nation_id if nation_id is not None else 0
                    }
                    break

        # Include all players that appear in the game (including barbarians, pirates, etc.)
        # Don't filter by is_alive so we can see all factions in charts
        return civilizations

    def collect_time_series(
        self,
        states: Dict[int, Dict],
        config: Any,
        civilizations: Dict[int, Dict]
    ) -> Dict[str, Dict[int, Dict[int, float]]]:
        """Collect all time series metrics

        Args:
            states: Dict mapping turn numbers to game states
            config: ReportConfig instance
            civilizations: Dict of civilization info

        Returns:
            Dict mapping metric_name to {turn: {player_id: value}}
        """
        sorted_turns = sorted(states.keys())
        player_ids = list(civilizations.keys())

        # Initialize data structures
        time_series = {
            "treasury": {},
            "population": {},
            "science": {},
            "territory_size": {},
            "arable_land": {},
            "food_production": {},
            "shield_production": {},
            "trade_production": {},
            "culture": {},
            "techs_known": {},
            "cities_count": {},
            "units_count": {}
        }

        # Collect data for each turn
        for turn in sorted_turns:
            state = states[turn]

            # Initialize turn data
            for metric in time_series:
                time_series[metric][turn] = {}

            if 'player' not in state:
                continue

            # Extract metrics for each player
            for pid_str in state['player'].keys():
                pid = int(pid_str)
                if pid not in player_ids:
                    continue

                player_info = state['player'][pid_str]
                if not isinstance(player_info, dict):
                    continue

                # Basic player metrics
                time_series["treasury"][turn][pid] = metrics.get_player_gold(state, pid)
                time_series["science"][turn][pid] = metrics.get_player_science_production(state, pid)
                time_series["culture"][turn][pid] = player_info.get('culture', 0)
                time_series["techs_known"][turn][pid] = metrics.count_known_techs(player_info)

                # Aggregated city metrics
                time_series["population"][turn][pid] = metrics.aggregate_city_metric(state, pid, 'size')
                time_series["food_production"][turn][pid] = metrics.aggregate_city_metric(state, pid, 'prod_food')
                time_series["shield_production"][turn][pid] = metrics.aggregate_city_metric(state, pid, 'prod_shield')
                time_series["trade_production"][turn][pid] = metrics.aggregate_city_metric(state, pid, 'prod_trade')

                # Territory metrics
                time_series["territory_size"][turn][pid] = metrics.calculate_territory_size(state, pid)
                time_series["arable_land"][turn][pid] = metrics.calculate_arable_land(state, pid)

                # Count cities and units
                cities = state.get('city', {})
                time_series["cities_count"][turn][pid] = len([
                    c for c in cities.values()
                    if isinstance(c, dict) and c.get('owner') == pid
                ])

                units = state.get('unit', {})
                time_series["units_count"][turn][pid] = len([
                    u for u in units.values()
                    if isinstance(u, dict) and u.get('owner') == pid
                ])

            # Try to override with complete savegame data
            savegame_data = get_savegame_data_for_report(config, turn)
            if savegame_data:
                # Override production data
                if 'production' in savegame_data:
                    for player_id, prod in savegame_data['production'].items():
                        if player_id in player_ids:
                            time_series["trade_production"][turn][player_id] = prod['trade']
                            time_series["food_production"][turn][player_id] = prod['food']
                            time_series["shield_production"][turn][player_id] = prod['shields']

                # Override science data
                if 'science' in savegame_data:
                    for player_id, sci in savegame_data['science'].items():
                        if player_id in player_ids:
                            time_series["science"][turn][player_id] = sci['science_per_turn']

                            # Only override techs_known if it's higher or equal
                            # Technologies can never decrease (can't un-discover a tech)
                            savegame_techs = sci['techs_known']
                            if player_id in time_series["techs_known"][turn]:
                                current_techs = time_series["techs_known"][turn][player_id]
                                if savegame_techs >= current_techs:
                                    time_series["techs_known"][turn][player_id] = savegame_techs
                                # Otherwise keep the state file value (it's more reliable)
                            else:
                                # No state data for this player, use savegame data
                                time_series["techs_known"][turn][player_id] = savegame_techs

        # Post-processing: Enforce monotonicity for techs_known
        # Technologies can never decrease - if we see a drop, it's bad data
        sorted_turns = sorted(time_series["techs_known"].keys())
        for pid in player_ids:
            max_techs_seen = 0
            for turn in sorted_turns:
                # Skip if this turn doesn't have data for this player
                if turn not in time_series["techs_known"]:
                    continue
                if pid in time_series["techs_known"][turn]:
                    current_techs = time_series["techs_known"][turn][pid]
                    # If current is less than max seen, use max instead (bad data)
                    if current_techs < max_techs_seen:
                        time_series["techs_known"][turn][pid] = max_techs_seen
                    else:
                        max_techs_seen = current_techs

        return time_series

    def collect_events(
        self,
        states: Dict[int, Dict],
        data_loader: Any,
        config: Any = None
    ) -> List[Dict[str, Any]]:
        """Collect all game events

        Args:
            states: Dict mapping turn numbers to game states
            data_loader: DataLoader instance
            config: Optional ReportConfig for savegame access

        Returns:
            List of event dictionaries
        """
        detector = EventDetector(data_loader)
        all_events = []

        sorted_turns = sorted(states.keys())
        prev_state = None
        prev_savegame_data = None

        # Track all techs seen so far for each player (to avoid duplicate discoveries)
        all_techs_seen = {}  # {player_id: set of tech_ids}

        for turn in sorted_turns:
            curr_state = states[turn]

            # Try to get savegame data for more complete tech visibility
            curr_savegame_data = None
            if config:
                curr_savegame_data = get_savegame_data_for_report(config, turn)

            # Detect regular events (city, government, diplomatic)
            game_events = detector.detect_city_events(prev_state, curr_state, turn)
            game_events.extend(detector.detect_government_changes(prev_state, curr_state, turn))
            game_events.extend(detector.detect_diplomatic_changes(prev_state, curr_state, turn))

            # For tech discoveries, prefer savegame data (complete visibility)
            # Use all_techs_seen to avoid counting same tech multiple times
            if curr_savegame_data:
                curr_techs = curr_savegame_data.get('technologies', {})

                # Get tech names from state if available
                tech_names = {}
                if 'tech' in curr_state:
                    for tech_id, tech_data in curr_state['tech'].items():
                        if isinstance(tech_data, dict):
                            tech_names[tech_id] = tech_data.get('name', f'Tech #{tech_id}')

                # Check each player for genuinely new techs
                for player_id, curr_player_techs in curr_techs.items():
                    if player_id not in all_techs_seen:
                        all_techs_seen[player_id] = set()

                    # Find techs this player knows that we haven't seen before
                    new_techs = curr_player_techs - all_techs_seen[player_id]

                    # Add these techs to the player's seen set
                    all_techs_seen[player_id].update(new_techs)

                    # Create discovery events
                    from ..utils.event_detector import GameEvent
                    for tech_id in new_techs:
                        tech_name = tech_names.get(tech_id, f'Tech #{tech_id}')
                        player_name = detector._get_player_name(curr_state, player_id)

                        game_events.append(GameEvent(
                            turn=turn,
                            event_type='tech_discovered',
                            description=f"{player_name} discovered {tech_name}",
                            player_id=player_id,
                            metadata={
                                'tech_id': tech_id,
                                'tech_name': tech_name
                            }
                        ))
            elif prev_state:
                # Fall back to state-based detection if savegame unavailable
                tech_events = detector.detect_tech_discoveries(prev_state, curr_state, turn)
                game_events.extend(tech_events)

            # Convert GameEvent objects to dictionaries
            for event in game_events:
                event_dict = {
                    "turn": event.turn,
                    "type": event.event_type,
                    "player_id": event.player_id,
                    "description": event.description,
                }

                if event.location:
                    event_dict["location"] = list(event.location)

                if event.metadata:
                    event_dict["metadata"] = event.metadata

                all_events.append(event_dict)

            prev_state = curr_state
            prev_savegame_data = curr_savegame_data

        return all_events

    def collect_snapshots(
        self,
        states: Dict[int, Dict],
        target_turn: int
    ) -> Dict[int, Dict[str, Any]]:
        """Collect snapshot data for specific turns

        Currently only collects for the final turn, but structure supports multiple snapshots.

        Args:
            states: Dict mapping turn numbers to game states
            target_turn: Turn to collect snapshot for (typically max turn)

        Returns:
            Dict mapping turn to snapshot data
        """
        snapshots = {}

        if target_turn not in states:
            return snapshots

        state = states[target_turn]
        players = state.get('player', {})

        # Collect scores and rankings
        scores = {}
        for pid_str, player_info in players.items():
            if isinstance(player_info, dict):
                pid = int(pid_str)
                scores[pid] = player_info.get('score', 0)

        # Create rankings
        rankings = []
        for rank, (pid, score) in enumerate(
            sorted(scores.items(), key=lambda x: x[1], reverse=True), 1
        ):
            rankings.append({
                "rank": rank,
                "player_id": pid,
                "score": score
            })

        # Count cities and units per player
        cities = state.get('city', {})
        units = state.get('unit', {})

        cities_count = {}
        units_count = {}

        for pid in scores.keys():
            cities_count[pid] = len([
                c for c in cities.values()
                if isinstance(c, dict) and c.get('owner') == pid
            ])
            units_count[pid] = len([
                u for u in units.values()
                if isinstance(u, dict) and u.get('owner') == pid
            ])

        # World totals
        total_cities = len([c for c in cities.values() if isinstance(c, dict)])
        total_units = len([u for u in units.values() if isinstance(u, dict)])
        total_population = sum(
            c.get('size', 0) for c in cities.values() if isinstance(c, dict)
        )

        snapshots[target_turn] = {
            "scores": scores,
            "rankings": rankings,
            "cities_count": cities_count,
            "units_count": units_count,
            "world_totals": {
                "total_cities": total_cities,
                "total_units": total_units,
                "total_population": total_population
            }
        }

        return snapshots

    # Helper methods (replicated from base_section.py)

    def _get_nation_name(self, nation_id: Optional[int], data_loader: Any) -> str:
        """Get nation/civilization name from nation ID

        Args:
            nation_id: Nation ID
            data_loader: DataLoader instance with ruleset data

        Returns:
            Nation name or empty string if not found
        """
        if nation_id is None:
            return ''

        if not data_loader.ruleset or 'nations' not in data_loader.ruleset:
            return ''

        nations = data_loader.ruleset['nations']
        nation_key = str(nation_id)

        if nation_key in nations:
            nation = nations[nation_key]
            return nation.get('adjective') or nation.get('rule_name', '')

        return ''

    def _get_nation_name_from_savegame(
        self,
        nation_identifier: Any,
        data_loader: Any
    ) -> str:
        """Get nation name from savegame nation identifier

        Args:
            nation_identifier: Nation ID (int) or name (str) from savegame
            data_loader: DataLoader instance with ruleset data

        Returns:
            Nation name or empty string if not found
        """
        if not data_loader.ruleset or 'nations' not in data_loader.ruleset:
            return ''

        nations = data_loader.ruleset['nations']

        # Try as numeric ID first
        try:
            nation_id = int(nation_identifier)
            nation_key = str(nation_id)
            if nation_key in nations:
                nation = nations[nation_key]
                return nation.get('adjective') or nation.get('rule_name', '')
        except (ValueError, TypeError):
            pass

        # Try as string name - search for matching rule_name
        nation_str = str(nation_identifier).strip('"')
        for nation_data in nations.values():
            if nation_data.get('rule_name') == nation_str:
                return nation_data.get('adjective') or nation_data.get('rule_name', '')

        return ''
