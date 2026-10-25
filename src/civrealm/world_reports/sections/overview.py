"""Section 1: Scenario Overview

Provides high-level summary of the game/world.
"""

from typing import Dict
from .base_section import BaseSection, SectionData


class OverviewSection(BaseSection):
    """Generate overview section"""

    def __init__(self):
        super().__init__('overview')

    def generate(
        self,
        states: Dict[int, Dict],
        config,
        data_loader,
        visualizer
    ) -> SectionData:
        """Generate overview section

        Args:
            states: Dict mapping turn numbers to game states
            config: ReportConfig instance
            data_loader: DataLoader instance
            visualizer: MapVisualizer instance

        Returns:
            SectionData with overview content
        """
        if not states:
            return SectionData(
                title='Scenario Overview',
                content='<h2>1. Scenario Overview</h2><p>No data available.</p>',
                images={},
                metadata={}
            )

        # Get first and last states
        turns = sorted(states.keys())
        first_turn = turns[0]
        last_turn = turns[-1]

        first_state = states[first_turn]
        last_state = states[last_turn]

        # Extract player information
        players = last_state.get('player', {})
        player_list = []

        for player_id in sorted(players.keys()):
            # Handle both string and int player IDs
            if isinstance(player_id, (int, str)):
                player = players[player_id]
                if isinstance(player, dict):
                    # Convert to int for consistent handling
                    pid_int = int(player_id) if isinstance(player_id, str) else player_id
                    # Use helper method to get name with civilization
                    name = self._get_player_name(pid_int, last_state, data_loader)
                    score = player.get('score', 0)
                    player_list.append((name, score))

        # Build HTML content
        html_parts = []
        html_parts.append('<h2>1. Scenario Overview</h2>')

        # Basic info
        html_parts.append('<h3>1.1 World Information</h3>')
        html_parts.append('<div class="overview-info">')
        html_parts.append(f'<p><strong>Turns Covered:</strong> {first_turn} to {last_turn} ({last_turn - first_turn + 1} turns)</p>')
        html_parts.append(f'<p><strong>Number of Civilizations:</strong> {len(player_list)}</p>')

        # Map info
        map_state = last_state.get('map', {})
        if map_state:
            xsize = map_state.get('xsize')
            ysize = map_state.get('ysize')
            if xsize is not None and ysize is not None:
                html_parts.append(f'<p><strong>Map Size:</strong> {xsize} × {ysize}</p>')

        html_parts.append('</div>')

        # Player list
        if player_list:
            html_parts.append('<h3>1.2 Civilizations</h3>')

            # Sort by score descending
            player_list.sort(key=lambda x: x[1], reverse=True)

            rows = []
            for rank, (name, score) in enumerate(player_list, 1):
                rows.append([rank, name, score])

            table = self._format_html_table(
                headers=['Rank', 'Civilization', 'Score'],
                rows=rows,
                caption=f'Civilizations at Turn {last_turn}'
            )
            html_parts.append(table)

        # Game statistics
        html_parts.append('<h3>1.3 World Statistics</h3>')

        stats = self._collect_statistics(last_state)
        html_parts.append('<div class="statistics">')
        html_parts.append('<ul>')
        for stat_name, stat_value in stats.items():
            html_parts.append(f'<li><strong>{stat_name}:</strong> {stat_value}</li>')
        html_parts.append('</ul>')
        html_parts.append('</div>')

        content = '\n'.join(html_parts)

        return SectionData(
            title='Scenario Overview',
            content=content,
            images={},
            metadata={
                'turns': len(turns),
                'players': len(player_list),
                'first_turn': first_turn,
                'last_turn': last_turn
            }
        )

    def _collect_statistics(self, state: Dict) -> Dict[str, any]:
        """Collect world statistics from state

        Args:
            state: Game state dict

        Returns:
            Dict of statistics
        """
        stats = {}

        # Count cities
        cities = state.get('city', {})
        stats['Total Cities'] = len([c for c in cities.values() if isinstance(c, dict)])

        # Count units
        units = state.get('unit', {})
        stats['Total Units'] = len([u for u in units.values() if isinstance(u, dict)])

        # Total population
        total_pop = 0
        for city in cities.values():
            if isinstance(city, dict):
                total_pop += city.get('size', 0)
        stats['Total Population'] = total_pop

        return stats
