"""Base class for report sections"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List
from io import BytesIO


@dataclass
class SectionData:
    """Data container for a report section

    Attributes:
        title: Section title
        content: HTML content for the section
        images: Dict mapping image names to BytesIO buffers
        metadata: Additional section metadata
    """
    title: str
    content: str
    images: Dict[str, BytesIO]
    metadata: Dict[str, Any]


class BaseSection(ABC):
    """Abstract base class for report sections

    All report sections should inherit from this class and implement
    the generate() method.
    """

    def __init__(self, section_name: str):
        """Initialize section

        Args:
            section_name: Unique identifier for this section
        """
        self.section_name = section_name

    @abstractmethod
    def generate(
        self,
        states: Dict[int, Dict],
        config: Any,
        data_loader: Any,
        visualizer: Any
    ) -> SectionData:
        """Generate section content

        Args:
            states: Dict mapping turn numbers to game states
            config: ReportConfig instance
            data_loader: DataLoader instance for accessing additional data
            visualizer: MapVisualizer instance for generating images

        Returns:
            SectionData containing the generated content and images
        """
        pass

    def _get_player_info(self, state: Dict) -> Dict[int, Dict]:
        """Extract player information from state

        Args:
            state: Game state dict

        Returns:
            Dict mapping player IDs to player info
        """
        return state.get('player', {})

    def _get_nation_name(self, nation_id: int, data_loader: Any) -> str:
        """Get nation/civilization name from nation ID

        Args:
            nation_id: Nation ID
            data_loader: DataLoader instance with ruleset data

        Returns:
            Nation name or empty string if not found
        """
        if data_loader.ruleset and 'nations' in data_loader.ruleset:
            nations = data_loader.ruleset['nations']
            nation_key = str(nation_id)  # Nations dict uses string keys
            if nation_key in nations:
                nation = nations[nation_key]
                # Try adjective first (e.g., "Abkhaz"), then rule_name as fallback
                return nation.get('adjective') or nation.get('rule_name', '')
        return ''

    def _get_nation_name_from_savegame_data(self, nation_identifier: any, data_loader: Any) -> str:
        """Get nation/civilization name from savegame nation identifier

        The savegame may store nation as either a numeric ID or a string name.

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

    def _get_player_name(self, player_id: int, state: Dict, data_loader: Any) -> str:
        """Get civilization name for a player

        Returns the civilization/nation name. Falls back to player name
        if civilization is unknown.

        Args:
            player_id: Player ID
            state: Game state dict
            data_loader: DataLoader instance with ruleset data

        Returns:
            Civilization name or player name as fallback
        """
        players = self._get_player_info(state)

        # Handle both int and string player IDs
        if isinstance(player_id, str):
            player_id = int(player_id)

        # Try to find player by int or string key
        player = None
        if player_id in players:
            player = players[player_id]
        elif str(player_id) in players:
            player = players[str(player_id)]

        if player and isinstance(player, dict):
            # Try to get civilization name first
            nation_id = player.get('nation')
            if nation_id is not None:
                civ_name = self._get_nation_name(nation_id, data_loader)
                if civ_name:
                    return civ_name

            # Fall back to player name
            return player.get('name', f'Player {player_id}')

        return f'Player {player_id}'

    def _collect_all_player_names(self, states: Dict[int, Dict], data_loader: Any) -> Dict[int, str]:
        """Collect civilization names for all players across all states

        This iterates through all states to find all players that appear at any turn,
        ensuring late-joining players are included in the player_names mapping.

        Args:
            states: Dict mapping turn numbers to game states
            data_loader: DataLoader instance with ruleset data

        Returns:
            Dict mapping player IDs to civilization names
        """
        player_names = {}
        all_player_ids = set()

        # First pass: collect all unique player IDs across all turns
        for state in states.values():
            if 'player' in state:
                for pid in state['player'].keys():
                    all_player_ids.add(int(pid))

        # Second pass: get names for each player (using latest state where they appear)
        for player_id in all_player_ids:
            # Find the latest state where this player exists
            for turn in sorted(states.keys(), reverse=True):
                state = states[turn]
                if 'player' in state:
                    pid_str = str(player_id)
                    if pid_str in state['player'] or player_id in state['player']:
                        player_names[player_id] = self._get_player_name(player_id, state, data_loader)
                        break

        return player_names

    def _filter_active_players(self, player_names: Dict[int, str], states: Dict[int, Dict]) -> Dict[int, str]:
        """Filter player_names to only include players who are alive in the final state

        Args:
            player_names: Dict mapping player_id to civilization name
            states: Dict mapping turn numbers to game states

        Returns:
            Filtered dict with only active players
        """
        if not states:
            return player_names

        # Get final state
        final_turn = max(states.keys())
        final_state = states[final_turn]

        if 'player' not in final_state:
            return player_names

        # Collect alive player IDs from final state
        alive_player_ids = set()
        for pid_str, player_info in final_state['player'].items():
            if isinstance(player_info, dict):
                if player_info.get('is_alive', False):
                    alive_player_ids.add(int(pid_str))

        # Filter player_names to only include alive players
        return {pid: name for pid, name in player_names.items() if pid in alive_player_ids}

    def _format_html_table(
        self,
        headers: List[str],
        rows: List[List[Any]],
        caption: str = ""
    ) -> str:
        """Format data as HTML table

        Args:
            headers: List of column headers
            rows: List of row data
            caption: Optional table caption

        Returns:
            HTML table string
        """
        html = ['<table class="data-table">']

        if caption:
            html.append(f'<caption>{caption}</caption>')

        # Headers
        html.append('<thead><tr>')
        for header in headers:
            html.append(f'<th>{header}</th>')
        html.append('</tr></thead>')

        # Rows
        html.append('<tbody>')
        for row in rows:
            html.append('<tr>')
            for cell in row:
                html.append(f'<td>{cell}</td>')
            html.append('</tr>')
        html.append('</tbody>')

        html.append('</table>')
        return '\n'.join(html)
