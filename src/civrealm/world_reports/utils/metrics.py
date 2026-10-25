"""Metrics calculation utilities for world reports

This module provides functions to calculate aggregated metrics from game state data.
"""

from typing import Dict, List, Any
import numpy as np


# Arable terrain type IDs (good for farming)
ARABLE_TERRAIN_IDS = [5, 6, 7, 8, 10, 11]  # Forest, Grassland, Hills, Jungle, Plains, Swamp


def calculate_territory_size(state: Dict, player_id: int) -> int:
    """Count total tiles controlled by a player

    Args:
        state: Game state dictionary containing 'map' with 'tile_owner'
        player_id: Player ID (as integer)

    Returns:
        Number of tiles controlled by the player
    """
    if 'map' not in state or 'tile_owner' not in state['map']:
        return 0

    tile_owner = np.array(state['map']['tile_owner'])
    return int(np.sum(tile_owner == player_id))


def calculate_arable_land(state: Dict, player_id: int) -> int:
    """Count arable (farmable) tiles controlled by a player

    Arable terrains are: Forest, Grassland, Hills, Jungle, Plains, Swamp

    Args:
        state: Game state dictionary containing 'map' with 'tile_owner' and 'terrain'
        player_id: Player ID (as integer)

    Returns:
        Number of arable tiles controlled by the player
    """
    if 'map' not in state:
        return 0

    map_state = state['map']
    if 'tile_owner' not in map_state or 'terrain' not in map_state:
        return 0

    tile_owner = np.array(map_state['tile_owner'])
    terrain = np.array(map_state['terrain'])

    # Find tiles owned by player
    player_tiles = tile_owner == player_id

    # Find arable tiles
    arable_mask = np.isin(terrain, ARABLE_TERRAIN_IDS)

    # Count tiles that are both owned by player AND arable
    return int(np.sum(player_tiles & arable_mask))


def aggregate_city_metric(state: Dict, player_id: int, metric_name: str) -> float:
    """Sum a metric across all cities owned by a player

    Args:
        state: Game state dictionary containing 'city' data
        player_id: Player ID (can be string or int)
        metric_name: Name of the metric to sum (e.g., 'prod_food', 'prod_shield', 'size')

    Returns:
        Sum of the metric across all cities owned by the player
    """
    if 'city' not in state:
        return 0.0

    # Convert player_id to string for comparison (state uses string keys)
    player_id_str = str(player_id)

    total = 0.0
    for city_id, city in state['city'].items():
        if not isinstance(city, dict):
            continue

        # Check if city is owned by player
        city_owner = str(city.get('owner', ''))
        if city_owner == player_id_str:
            # Add metric value (default to 0 if not present)
            value = city.get(metric_name, 0)
            # Handle negative values (sometimes cities show -1 for missing data)
            if value >= 0:
                total += value

    return float(total)


def count_known_techs(player_state: Dict) -> int:
    """Count technologies known by a player

    Technologies are stored as tech_1, tech_2, ..., tech_87 with value 18 = TECH_KNOWN

    Args:
        player_state: Player state dictionary

    Returns:
        Number of known technologies
    """
    if not isinstance(player_state, dict):
        return 0

    count = 0
    for key, value in player_state.items():
        # Check if key is tech_N format
        if key.startswith('tech_') and key[5:].isdigit():
            # 18 is the TECH_KNOWN state
            if value == 18:
                count += 1

    return count


def aggregate_happiness(state: Dict, player_id: int) -> Dict[str, int]:
    """Aggregate happiness data across all cities owned by a player

    Args:
        state: Game state dictionary containing 'city' data
        player_id: Player ID (can be string or int)

    Returns:
        Dictionary with keys: 'happy', 'content', 'unhappy', 'angry'
    """
    if 'city' not in state:
        return {'happy': 0, 'content': 0, 'unhappy': 0, 'angry': 0}

    # Convert player_id to string for comparison
    player_id_str = str(player_id)

    totals = {'happy': 0, 'content': 0, 'unhappy': 0, 'angry': 0}

    for city_id, city in state['city'].items():
        if not isinstance(city, dict):
            continue

        # Check if city is owned by player
        city_owner = str(city.get('owner', ''))
        if city_owner == player_id_str:
            # Add happiness values (only if valid - skip -1 values)
            for key in totals.keys():
                city_key = f'ppl_{key}'
                value = city.get(city_key, 0)
                if value >= 0:
                    totals[key] += value

    return totals


def get_player_science_production(state: Dict, player_id: int) -> float:
    """Get science production per turn for a player

    Args:
        state: Game state dictionary containing 'player' data
        player_id: Player ID (can be string or int)

    Returns:
        Science production per turn
    """
    if 'player' not in state:
        return 0.0

    # Convert player_id to string for dict lookup
    player_id_str = str(player_id)

    if player_id_str not in state['player']:
        return 0.0

    player = state['player'][player_id_str]
    if not isinstance(player, dict):
        return 0.0

    return float(player.get('science', 0))


def get_player_gold(state: Dict, player_id: int) -> float:
    """Get gold treasury for a player

    Args:
        state: Game state dictionary containing 'player' data
        player_id: Player ID (can be string or int)

    Returns:
        Gold treasury amount
    """
    if 'player' not in state:
        return 0.0

    # Convert player_id to string for dict lookup
    player_id_str = str(player_id)

    if player_id_str not in state['player']:
        return 0.0

    player = state['player'][player_id_str]
    if not isinstance(player, dict):
        return 0.0

    return float(player.get('gold', 0))


def get_player_culture(state: Dict, player_id: int) -> float:
    """Get culture points for a player

    Args:
        state: Game state dictionary containing 'player' data
        player_id: Player ID (can be string or int)

    Returns:
        Culture points
    """
    if 'player' not in state:
        return 0.0

    # Convert player_id to string for dict lookup
    player_id_str = str(player_id)

    if player_id_str not in state['player']:
        return 0.0

    player = state['player'][player_id_str]
    if not isinstance(player, dict):
        return 0.0

    return float(player.get('culture', 0))
