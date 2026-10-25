"""Section 2: Major Historical Events

Tracks city founding, conquests, and other major events throughout the game.
"""

from typing import Dict, List
from .base_section import BaseSection, SectionData
from ..utils.event_detector import EventDetector, GameEvent
from ..utils import metrics, graphs


def select_snapshot_turns(max_turn: int, max_snapshots: int = 5,
                         min_turn: int = 5, min_spacing: int = 10) -> List[int]:
    """
    Select snapshot turns for territorial maps using backward selection with adaptive spacing.

    Args:
        max_turn: The latest turn in the report
        max_snapshots: Maximum number of snapshots to include (default: 5)
        min_turn: Earliest turn to consider (default: 5, avoids empty early game)
        min_spacing: Minimum turns between snapshots to avoid clustering (default: 10)

    Strategy:
    - Start from max_turn (most recent)
    - Work backwards with adaptive spacing
    - Spacing = max(min_spacing, available_range / (max_snapshots - 1))
      This ensures:
        * Long games: wider spacing (e.g., 500 turns → ~100 turn gaps)
        * Short games: minimum spacing (e.g., 40 turns → 10 turn gaps)
        * Very short games: fewer maps returned (acceptable per requirements)
    - Stops when reaching min_turn or max_snapshots limit

    Examples:
        max_turn=100 → [12, 35, 58, 81, 100] (spacing ~23)
        max_turn=50  → [6, 17, 28, 39, 50]   (spacing ~11)
        max_turn=30  → [10, 20, 30]          (spacing 10, only 3 maps - acceptable)
        max_turn=15  → [15]                  (only 1 map - game too short)

    Returns:
        Sorted list of turn numbers for snapshot generation
    """
    # Edge cases: game too short
    if max_turn < min_turn:
        return []

    if max_turn == min_turn:
        return [min_turn]

    # Calculate adaptive spacing
    # available_range: number of turns between min and max
    available_range = max_turn - min_turn

    # Ideal spacing to fit max_snapshots evenly across the range
    # Example: 95 turns with 5 snapshots → 95/4 = 23 turns apart
    ideal_spacing = available_range // (max_snapshots - 1) if max_snapshots > 1 else available_range

    # Use the larger of min_spacing or ideal_spacing
    # This ensures we don't cluster maps too closely in short games
    spacing = max(min_spacing, ideal_spacing)

    # Build snapshot list working backwards from max_turn
    snapshots = [max_turn]
    current = max_turn

    while len(snapshots) < max_snapshots:
        next_turn = current - spacing

        if next_turn < min_turn:
            # Check if we can squeeze in one more snapshot at min_turn
            if min_turn not in snapshots and (current - min_turn) >= min_spacing:
                snapshots.append(min_turn)
            break

        snapshots.append(next_turn)
        current = next_turn

    return sorted(snapshots)


class HistoricalEventsSection(BaseSection):
    """Generate historical events section"""

    def __init__(self):
        super().__init__('historical_events')

    def generate(
        self,
        states: Dict[int, Dict],
        config,
        data_loader,
        visualizer
    ) -> SectionData:
        """Generate historical events section

        Args:
            states: Dict mapping turn numbers to game states
            config: ReportConfig instance
            data_loader: DataLoader instance
            visualizer: MapVisualizer instance

        Returns:
            SectionData with events table and mini-maps
        """
        detector = EventDetector(data_loader)
        all_events: List[GameEvent] = []

        # Sort turns
        sorted_turns = sorted(states.keys())

        # Detect events by comparing consecutive states
        prev_state = None
        for turn in sorted_turns:
            curr_state = states[turn]
            events = detector.detect_all_events(prev_state, curr_state, turn)
            all_events.extend(events)
            prev_state = curr_state

        # Generate content
        images = {}

        # Section 2.1: City Events
        city_events = [e for e in all_events if e.event_type in
                      ['city_founded', 'city_conquered', 'city_destroyed']]

        html_parts = []
        html_parts.append('<h2>2. Major Historical Events</h2>')
        html_parts.append('<h3>2.1 City Founding and Conquest</h3>')

        if city_events:
            # Create events table
            table_rows = []
            for event in city_events:
                turn_str = f"Turn {event.turn}"
                event_type = event.event_type.replace('_', ' ').title()
                description = event.description

                # Add location if available
                if event.location:
                    x, y = event.location
                    description += f" at ({x}, {y})"

                table_rows.append([turn_str, event_type, description])

            events_table = self._format_html_table(
                headers=['Turn', 'Event Type', 'Description'],
                rows=table_rows,
                caption='City Events Throughout History'
            )
            html_parts.append(events_table)
        else:
            html_parts.append('<p>No city events recorded.</p>')

        # Section 2.2: Territorial Control (if we have map data)
        if states:
            html_parts.append('<h3>2.2 Territorial Control</h3>')

            # Generate territory maps at key turns using adaptive selection
            max_turn = max(states.keys())
            snapshot_turns = select_snapshot_turns(
                max_turn=max_turn,
                max_snapshots=5,
                min_turn=5,
                min_spacing=10
            )
            # Filter to only turns we have state data for
            snapshot_turns = [t for t in snapshot_turns if t in states]

            if snapshot_turns:
                html_parts.append('<div class="territory-maps">')

                for turn in snapshot_turns:
                    state = states[turn]
                    img_name = f'territory_turn{turn}'

                    try:
                        img_buf = visualizer.render_territory_map(
                            map_state=state.get('map', {}),
                            player_state=state.get('player', {}),
                            title=f"Territorial Control - Turn {turn}",
                            highlight_cities=True,
                            show_legend=True
                        )
                        images[img_name] = img_buf

                        html_parts.append(f'<div class="territory-map">')
                        html_parts.append(f'<img src="{img_name}.png" alt="Territory at turn {turn}"/>')
                        html_parts.append('</div>')
                    except Exception as e:
                        print(f"Warning: Failed to generate territory map for turn {turn}: {e}")

                html_parts.append('</div>')

            # Generate territory and arable land graphs
            html_parts.append('<h4>Territorial Expansion Over Time</h4>')

            # Get player names (with civilization names)
            first_state = states[sorted_turns[0]] if sorted_turns else {}
            player_names = {}
            if 'player' in first_state:
                for pid, player in first_state['player'].items():
                    if isinstance(player, dict):
                        pid_int = int(pid)
                        player_names[pid_int] = self._get_player_name(pid_int, first_state, data_loader)

            # Calculate territory data for all turns
            territory_data = {}  # {turn: {player_id: tiles}}
            arable_data = {}     # {turn: {player_id: arable_tiles}}

            for turn in sorted_turns:
                state = states[turn]
                territory_data[turn] = {}
                arable_data[turn] = {}

                if 'player' in state:
                    for pid_str in state['player'].keys():
                        pid = int(pid_str)
                        territory_data[turn][pid] = metrics.calculate_territory_size(state, pid)
                        arable_data[turn][pid] = metrics.calculate_arable_land(state, pid)

            # Generate territory size graph
            if territory_data:
                try:
                    img_buf = graphs.create_time_series_graph(
                        data=territory_data,
                        title='Territorial Extent Over Time',
                        ylabel='Tiles Controlled',
                        player_names=player_names,
                        dpi=config.dpi if hasattr(config, 'dpi') else 150
                    )
                    img_name = 'territory_size_over_time'
                    images[img_name] = img_buf
                    html_parts.append(f'<div class="graph">')
                    html_parts.append(f'<img src="{img_name}.png" alt="Territory size over time"/>')
                    html_parts.append('</div>')
                except Exception as e:
                    print(f"Warning: Failed to generate territory size graph: {e}")

            # Generate arable land graph
            if arable_data:
                try:
                    img_buf = graphs.create_time_series_graph(
                        data=arable_data,
                        title='Arable Land Controlled Over Time',
                        ylabel='Arable Tiles Controlled',
                        player_names=player_names,
                        dpi=config.dpi if hasattr(config, 'dpi') else 150
                    )
                    img_name = 'arable_land_over_time'
                    images[img_name] = img_buf
                    html_parts.append(f'<div class="graph">')
                    html_parts.append(f'<img src="{img_name}.png" alt="Arable land over time"/>')
                    html_parts.append('</div>')
                except Exception as e:
                    print(f"Warning: Failed to generate arable land graph: {e}")

        # Technology and government events
        tech_events = [e for e in all_events if e.event_type == 'tech_discovered']
        gov_events = [e for e in all_events if e.event_type == 'government_change']

        if tech_events:
            html_parts.append('<h3>2.3 Major Technological Advances</h3>')
            tech_rows = []
            for event in tech_events[:20]:  # Limit to first 20
                turn_str = f"Turn {event.turn}"
                player_name = self._get_player_name(event.player_id, states.get(event.turn, {}), data_loader)
                tech_rows.append([turn_str, player_name, event.description])

            tech_table = self._format_html_table(
                headers=['Turn', 'Civilization', 'Discovery'],
                rows=tech_rows,
                caption='Major Technological Discoveries'
            )
            html_parts.append(tech_table)

        if gov_events:
            html_parts.append('<h3>2.4 Government Changes</h3>')
            gov_rows = []
            for event in gov_events:
                turn_str = f"Turn {event.turn}"
                gov_rows.append([turn_str, event.description])

            gov_table = self._format_html_table(
                headers=['Turn', 'Change'],
                rows=gov_rows,
                caption='Government Transitions'
            )
            html_parts.append(gov_table)

        content = '\n'.join(html_parts)

        return SectionData(
            title='Major Historical Events',
            content=content,
            images=images,
            metadata={
                'total_events': len(all_events),
                'city_events': len(city_events),
                'tech_events': len(tech_events),
                'gov_events': len(gov_events)
            }
        )
