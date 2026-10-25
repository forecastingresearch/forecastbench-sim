"""Event detection from game state changes"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GameEvent:
    """Represents a game event

    Attributes:
        turn: Turn number when event occurred
        event_type: Type of event (city_founded, city_conquered, etc.)
        description: Human-readable description
        player_id: Primary player involved
        location: Optional (x, y) coordinates
        metadata: Additional event-specific data
    """
    turn: int
    event_type: str
    description: str
    player_id: int
    location: Optional[Tuple[int, int]] = None
    metadata: Optional[Dict] = None


class EventDetector:
    """Detect events by comparing consecutive game states"""

    def __init__(self, data_loader=None):
        """Initialize event detector

        Args:
            data_loader: Optional DataLoader instance for accessing ruleset data
        """
        self.events: List[GameEvent] = []
        self.data_loader = data_loader

    def detect_city_events(
        self,
        prev_state: Optional[Dict],
        curr_state: Dict,
        turn: int
    ) -> List[GameEvent]:
        """Detect city founding, conquest, and destruction events

        Args:
            prev_state: Previous turn state (None if first turn)
            curr_state: Current turn state
            turn: Current turn number

        Returns:
            List of detected city events
        """
        events = []

        curr_cities = curr_state.get('city', {})

        if prev_state is None:
            # First turn - all cities are "founded"
            for city_id, city in curr_cities.items():
                if isinstance(city, dict):
                    events.append(self._create_city_founded_event(
                        city_id, city, curr_state, turn, is_initial=True
                    ))
            return events

        prev_cities = prev_state.get('city', {})

        # Detect new cities (founded)
        new_city_ids = set(curr_cities.keys()) - set(prev_cities.keys())
        for city_id in new_city_ids:
            city = curr_cities[city_id]
            if isinstance(city, dict):
                events.append(self._create_city_founded_event(
                    city_id, city, curr_state, turn, is_initial=False
                ))

        # Detect conquered cities (owner changed)
        for city_id in set(curr_cities.keys()) & set(prev_cities.keys()):
            prev_city = prev_cities[city_id]
            curr_city = curr_cities[city_id]

            if isinstance(prev_city, dict) and isinstance(curr_city, dict):
                prev_owner = prev_city.get('owner')
                curr_owner = curr_city.get('owner')

                if prev_owner is not None and curr_owner is not None and prev_owner != curr_owner:
                    events.append(self._create_city_conquered_event(
                        city_id, prev_city, curr_city, prev_state, curr_state, turn
                    ))

        # Detect destroyed cities (disappeared)
        destroyed_city_ids = set(prev_cities.keys()) - set(curr_cities.keys())
        for city_id in destroyed_city_ids:
            city = prev_cities[city_id]
            if isinstance(city, dict):
                events.append(self._create_city_destroyed_event(
                    city_id, city, prev_state, turn
                ))

        return events

    def detect_tech_discoveries(
        self,
        prev_state: Optional[Dict],
        curr_state: Dict,
        turn: int
    ) -> List[GameEvent]:
        """Detect technology discoveries

        Args:
            prev_state: Previous turn state (None if first turn)
            curr_state: Current turn state
            turn: Current turn number

        Returns:
            List of tech discovery events
        """
        events = []

        if prev_state is None:
            return events

        # Compare player tech flags
        prev_players = prev_state.get('player', {})
        curr_players = curr_state.get('player', {})

        for player_id in curr_players:
            if player_id not in prev_players:
                continue

            prev_player = prev_players[player_id]
            curr_player = curr_players[player_id]

            if not isinstance(prev_player, dict) or not isinstance(curr_player, dict):
                continue

            # Look for tech_N flags that changed from False to True
            for key, value in curr_player.items():
                if key.startswith('tech_') and value is True:
                    prev_value = prev_player.get(key, False)
                    if not prev_value:
                        # Tech discovered!
                        tech_num = key.split('_')[1]
                        events.append(GameEvent(
                            turn=turn,
                            event_type='tech_discovered',
                            description=f"Technology #{tech_num} discovered",
                            player_id=player_id,
                            metadata={'tech_id': tech_num}
                        ))

        return events

    def detect_government_changes(
        self,
        prev_state: Optional[Dict],
        curr_state: Dict,
        turn: int
    ) -> List[GameEvent]:
        """Detect government/policy changes

        Args:
            prev_state: Previous turn state (None if first turn)
            curr_state: Current turn state
            turn: Current turn number

        Returns:
            List of government change events
        """
        events = []

        if prev_state is None:
            return events

        prev_players = prev_state.get('player', {})
        curr_players = curr_state.get('player', {})

        for player_id in curr_players:
            if player_id not in prev_players:
                continue

            prev_player = prev_players[player_id]
            curr_player = curr_players[player_id]

            if not isinstance(prev_player, dict) or not isinstance(curr_player, dict):
                continue

            prev_gov = prev_player.get('government_name', '')
            curr_gov = curr_player.get('government_name', '')

            if prev_gov and curr_gov and prev_gov != curr_gov:
                player_name = curr_player.get('name', f'Player {player_id}')
                events.append(GameEvent(
                    turn=turn,
                    event_type='government_change',
                    description=f"{player_name} changed government from {prev_gov} to {curr_gov}",
                    player_id=player_id,
                    metadata={'from': prev_gov, 'to': curr_gov}
                ))

        return events

    def detect_diplomatic_changes(
        self,
        prev_state: Optional[Dict],
        curr_state: Dict,
        turn: int
    ) -> List[GameEvent]:
        """Detect wars, peace, and other diplomatic changes

        Args:
            prev_state: Previous turn state (None if first turn)
            curr_state: Current turn state
            turn: Current turn number

        Returns:
            List of diplomatic events
        """
        events = []

        if prev_state is None:
            return events

        prev_dipl = prev_state.get('dipl', {})
        curr_dipl = curr_state.get('dipl', {})

        # Compare diplomatic relationships
        for key in curr_dipl:
            if key not in prev_dipl:
                continue

            prev_rel = prev_dipl[key]
            curr_rel = curr_dipl[key]

            if not isinstance(prev_rel, dict) or not isinstance(curr_rel, dict):
                continue

            # Check for changes in diplomatic state
            prev_state_val = prev_rel.get('state', '')
            curr_state_val = curr_rel.get('state', '')

            if prev_state_val != curr_state_val:
                player1 = curr_rel.get('player1', 0)
                player2 = curr_rel.get('player2', 0)

                events.append(GameEvent(
                    turn=turn,
                    event_type='diplomatic_change',
                    description=f"Diplomatic state between Player {player1} and Player {player2} changed from {prev_state_val} to {curr_state_val}",
                    player_id=player1,
                    metadata={
                        'player1': player1,
                        'player2': player2,
                        'from_state': prev_state_val,
                        'to_state': curr_state_val
                    }
                ))

        return events

    def detect_all_events(
        self,
        prev_state: Optional[Dict],
        curr_state: Dict,
        turn: int
    ) -> List[GameEvent]:
        """Detect all types of events

        Args:
            prev_state: Previous turn state (None if first turn)
            curr_state: Current turn state
            turn: Current turn number

        Returns:
            List of all detected events
        """
        events = []
        events.extend(self.detect_city_events(prev_state, curr_state, turn))
        events.extend(self.detect_tech_discoveries(prev_state, curr_state, turn))
        events.extend(self.detect_government_changes(prev_state, curr_state, turn))
        events.extend(self.detect_diplomatic_changes(prev_state, curr_state, turn))
        return events

    def _get_player_name(self, state: Dict, player_id: int) -> str:
        """Get player civilization name from state

        Args:
            state: Game state dictionary
            player_id: Player ID (as int or str)

        Returns:
            Civilization name, falling back to player name if not found
        """
        if not state or 'player' not in state:
            return f'Player {player_id}'

        players = state['player']
        player_id_str = str(player_id)

        if player_id_str in players:
            player = players[player_id_str]
            if isinstance(player, dict):
                # Try to get civilization name first
                if self.data_loader and self.data_loader.ruleset:
                    nation_id = player.get('nation')
                    if nation_id is not None and 'nations' in self.data_loader.ruleset:
                        nations = self.data_loader.ruleset['nations']
                        nation_key = str(nation_id)
                        if nation_key in nations:
                            nation = nations[nation_key]
                            # Try adjective first, then rule_name
                            civ_name = nation.get('adjective') or nation.get('rule_name')
                            if civ_name:
                                return civ_name

                # Fall back to player name
                return player.get('name', f'Player {player_id}')

        return f'Player {player_id}'

    def _create_city_founded_event(
        self,
        city_id: str,
        city: Dict,
        state: Dict,
        turn: int,
        is_initial: bool
    ) -> GameEvent:
        """Create a city founded event"""
        city_name = city.get('name', f'City {city_id}')
        owner = city.get('owner', 0)
        owner_name = self._get_player_name(state, owner)
        x = city.get('x', 0)
        y = city.get('y', 0)

        if is_initial:
            description = f"Initial city: {city_name} ({owner_name})"
        else:
            description = f"{city_name} founded by {owner_name}"

        return GameEvent(
            turn=turn,
            event_type='city_founded',
            description=description,
            player_id=owner,
            location=(x, y),
            metadata={'city_id': city_id, 'city_name': city_name}
        )

    def _create_city_conquered_event(
        self,
        city_id: str,
        prev_city: Dict,
        curr_city: Dict,
        prev_state: Dict,
        curr_state: Dict,
        turn: int
    ) -> GameEvent:
        """Create a city conquered event"""
        city_name = curr_city.get('name', f'City {city_id}')
        prev_owner = prev_city.get('owner', 0)
        new_owner = curr_city.get('owner', 0)
        prev_owner_name = self._get_player_name(prev_state, prev_owner)
        new_owner_name = self._get_player_name(curr_state, new_owner)
        x = curr_city.get('x', 0)
        y = curr_city.get('y', 0)

        description = f"{city_name} conquered by {new_owner_name} from {prev_owner_name}"

        return GameEvent(
            turn=turn,
            event_type='city_conquered',
            description=description,
            player_id=new_owner,
            location=(x, y),
            metadata={
                'city_id': city_id,
                'city_name': city_name,
                'prev_owner': prev_owner,
                'new_owner': new_owner
            }
        )

    def _create_city_destroyed_event(
        self,
        city_id: str,
        city: Dict,
        state: Dict,
        turn: int
    ) -> GameEvent:
        """Create a city destroyed event"""
        city_name = city.get('name', f'City {city_id}')
        owner = city.get('owner', 0)
        owner_name = self._get_player_name(state, owner)
        x = city.get('x', 0)
        y = city.get('y', 0)

        description = f"{city_name} ({owner_name}) was destroyed"

        return GameEvent(
            turn=turn,
            event_type='city_destroyed',
            description=description,
            player_id=owner,
            location=(x, y),
            metadata={'city_id': city_id, 'city_name': city_name}
        )
