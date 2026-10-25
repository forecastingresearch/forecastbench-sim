"""Parser for Freeciv savegame files

This module extracts complete game state from Freeciv .sav files,
which contain data for ALL players without fog-of-war limitations.
"""

import re
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess


def list_server_savegames(username: str, host: str = 'localhost', port: int = 8080) -> List[str]:
    """List savegames on the Freeciv server

    Args:
        username: Player username
        host: Server host
        port: Server port

    Returns:
        List of savegame names
    """
    try:
        url = f"http://{host}:{port}/listsavegames?username={username}"
        response = requests.post(url, timeout=5)
        if response.status_code == 200 and response.text:
            return [s.strip() for s in response.text.split(';') if s.strip()]
    except Exception as e:
        print(f"Warning: Could not list server savegames: {e}")
    return []


def download_savegame(savegame_name: str, username: str, host: str = 'localhost', port: int = 8080, container_name: str = 'freeciv-web') -> Optional[tuple]:
    """Download savegame from server

    Args:
        savegame_name: Name of the savegame (may or may not include .sav.xz extension)
        username: Player username
        host: Server host
        port: Server port
        container_name: Docker container name

    Returns:
        Tuple of (content_bytes, actual_filename) if successful, None otherwise
    """
    try:
        # Savegames are stored in Docker container, not accessible via HTTP
        # Use docker exec to read the file
        # First, find the actual filename (may have .sav.xz or .sav.zst extension)
        base_path = f"/var/lib/tomcat10/webapps/data/savegames/{username}"

        # List files matching the savegame name
        list_cmd = ['docker', 'exec', container_name, 'ls', base_path]
        result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print(f"Warning: Could not list savegames in container: {result.stderr}")
            return None

        # Find matching filename (may have .sav.xz or .sav.zst extension)
        files = result.stdout.strip().split('\n')
        matching_file = None
        for f in files:
            if f == savegame_name or f.startswith(savegame_name + '.sav'):
                matching_file = f
                break

        if not matching_file:
            print(f"Warning: Could not find savegame file matching {savegame_name}")
            return None

        # Read the file from container
        file_path = f"{base_path}/{matching_file}"
        cat_cmd = ['docker', 'exec', container_name, 'cat', file_path]
        result = subprocess.run(cat_cmd, capture_output=True, timeout=10)

        if result.returncode == 0:
            return (result.stdout, matching_file)
        else:
            print(f"Warning: Could not read savegame {matching_file}: {result.stderr.decode('utf-8')}")
            return None

    except Exception as e:
        print(f"Warning: Could not download savegame {savegame_name}: {e}")
    return None


def find_latest_savegame(username: str, turn: int, host: str = 'localhost', port: int = 8080) -> Optional[str]:
    """Find the savegame file for a specific turn

    Args:
        username: Player username
        turn: Turn number
        host: Server host
        port: Server port

    Returns:
        Savegame name if found, None otherwise
    """
    # List savegames from server
    savegames = list_server_savegames(username, host, port)

    # Filter for the specific turn
    # Savegames are named like: {username}_T{turn}_{timestamp}.sav(.zst|.xz)?
    pattern = f"{username}_T{turn}_"
    matches = [s for s in savegames if s.startswith(pattern)]

    if matches:
        # Return the most recent (last in list, as they're chronologically ordered)
        return matches[-1]

    return None


def decompress_savegame_content(savegame_content: bytes, filename: str) -> str:
    """Decompress savegame content

    Args:
        savegame_content: Raw savegame bytes
        filename: Filename (used to determine compression)

    Returns:
        Decompressed savegame content as string
    """
    if filename.endswith('.zst'):
        # Use zstd to decompress
        result = subprocess.run(
            ['zstd', '-d', '-c'],
            input=savegame_content,
            capture_output=True,
            check=True
        )
        return result.stdout.decode('utf-8')
    elif filename.endswith('.xz'):
        # Use xz to decompress
        result = subprocess.run(
            ['xz', '-d', '-c'],
            input=savegame_content,
            capture_output=True,
            check=True
        )
        return result.stdout.decode('utf-8')
    else:
        # Uncompressed
        return savegame_content.decode('utf-8')


def get_local_savegames_dir(recording_dir: str) -> Path:
    """Get the local savegames directory path within the recordings directory

    Args:
        recording_dir: Path to the recordings directory (e.g., logs/recordings/username/)

    Returns:
        Path to the savegames subdirectory
    """
    savegames_dir = Path(recording_dir) / 'savegames'
    savegames_dir.mkdir(exist_ok=True)
    return savegames_dir


def find_local_savegame_for_turn(username: str, turn: int, recording_dir: str) -> Optional[str]:
    """Find a locally cached savegame for a specific turn

    Args:
        username: Player username
        turn: Turn number
        recording_dir: Path to the recordings directory

    Returns:
        Savegame filename if found, None otherwise
    """
    savegames_dir = get_local_savegames_dir(recording_dir)

    # Pattern: username_T{turn}_*.sav*
    # Try both turn and turn+1 (as final save after turn N may be labeled as N+1)
    for t in [turn, turn + 1]:
        pattern = f"{username}_T{t}_*"
        matches = list(savegames_dir.glob(pattern))
        if matches:
            # Return the most recent (sort by name, which includes timestamp)
            return sorted(matches)[-1].name

    return None


def load_local_savegame(savegame_name: str, recording_dir: str) -> Optional[Tuple[bytes, str]]:
    """Load a savegame from the local recordings directory

    Args:
        savegame_name: Name of the savegame file
        recording_dir: Path to the recordings directory

    Returns:
        Tuple of (content_bytes, filename) if found, None otherwise
    """
    savegames_dir = get_local_savegames_dir(recording_dir)

    # Try to find matching file (may have different extensions)
    for ext in ['', '.sav', '.sav.xz', '.sav.zst']:
        # Try exact match first
        filepath = savegames_dir / f"{savegame_name}{ext}"
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return (f.read(), filepath.name)

        # Also try without the extension if savegame_name already has one
        if savegame_name.endswith('.sav'):
            base_name = savegame_name.rsplit('.sav', 1)[0]
            filepath = savegames_dir / f"{base_name}{ext}"
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    return (f.read(), filepath.name)

    return None


def save_local_savegame(savegame_bytes: bytes, filename: str, recording_dir: str) -> None:
    """Save a savegame to the local recordings directory

    Args:
        savegame_bytes: Raw savegame bytes
        filename: Filename to save as
        recording_dir: Path to the recordings directory
    """
    savegames_dir = get_local_savegames_dir(recording_dir)
    filepath = savegames_dir / filename

    with open(filepath, 'wb') as f:
        f.write(savegame_bytes)

    print(f"Saved savegame to {filepath}")


def download_all_savegames_from_docker(username: str, recording_dir: str, container_name: str = 'freeciv-web') -> tuple[int, int, int]:
    """Download all savegames directly from Docker container

    This function lists all savegame files in the Docker container and downloads
    them to the local recordings directory. It skips files that already exist locally.

    Args:
        username: Player username
        recording_dir: Path to recordings directory
        container_name: Docker container name (default: 'freeciv-web')

    Returns:
        Tuple of (downloaded_count, skipped_count, failed_count)
    """
    # Create savegames directory
    savegames_dir = get_local_savegames_dir(recording_dir)

    # List files in Docker container
    docker_path = f"/var/lib/tomcat10/webapps/data/savegames/{username}"

    list_cmd = ['docker', 'exec', container_name, 'ls', docker_path]
    result = subprocess.run(list_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error listing Docker savegames: {result.stderr}")
        return (0, 0, 0)

    files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]

    if not files:
        print(f"No savegames found in Docker container")
        return (0, 0, 0)

    # Download each file
    downloaded = 0
    skipped = 0
    failed = 0

    for filename in files:
        # Skip if already downloaded
        local_file = savegames_dir / filename
        if local_file.exists():
            skipped += 1
            continue

        # Read file from container
        file_path = f"{docker_path}/{filename}"
        cat_cmd = ['docker', 'exec', container_name, 'cat', file_path]
        result = subprocess.run(cat_cmd, capture_output=True)

        if result.returncode == 0:
            # Save to local directory
            with open(local_file, 'wb') as f:
                f.write(result.stdout)
            downloaded += 1
        else:
            failed += 1

    return (downloaded, skipped, failed)


def parse_city_production(savegame_content: str) -> Dict[int, Dict[str, float]]:
    """Parse city production data from savegame

    The savegame contains complete production data for all players' cities,
    bypassing the fog-of-war limitations in client recordings.

    Args:
        savegame_content: Decompressed savegame file content

    Returns:
        Dict mapping player_id to aggregated production:
        {
            player_id: {
                'food': estimated_food_production,
                'shields': actual_shield_production,
                'trade': estimated_trade_production
            }
        }
    """
    production = {}

    # Find all player sections
    player_sections = re.finditer(r'\[player(\d+)\]', savegame_content)

    for player_match in player_sections:
        player_id = int(player_match.group(1))
        player_start = player_match.end()

        # Find the next section (or end of file)
        next_section = savegame_content.find('\n[', player_start)
        if next_section == -1:
            player_content = savegame_content[player_start:]
        else:
            player_content = savegame_content[player_start:next_section]

        # Find city schema definition
        # c={"y","x","id",...,"size","food_stock","shield_stock","last_turns_shield_surplus",...}
        schema_match = re.search(r'c=\{([^}]+)\}', player_content)
        if not schema_match:
            continue

        # Parse schema to find indices
        schema = schema_match.group(1).replace('"', '').split(',')

        # Find indices for production metrics
        try:
            size_idx = schema.index('size')
            shield_surplus_idx = schema.index('last_turns_shield_surplus')
        except ValueError:
            continue

        # Extract city data lines
        # Cities are stored as comma-separated values after the schema
        # Find lines that look like city data (start with coordinates)
        city_lines = re.findall(r'\n(\d+,\d+,\d+,[^\n]+)', player_content)

        total_food = 0.0
        total_shields = 0.0
        total_trade = 0.0

        for city_line in city_lines:
            # Parse CSV carefully, handling quoted strings
            values = []
            current = ""
            in_quotes = False
            for char in city_line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    values.append(current)
                    current = ""
                else:
                    current += char
            values.append(current)  # Add last value

            if len(values) <= max(size_idx, shield_surplus_idx):
                continue

            try:
                # Get city size
                city_size = int(values[size_idx])

                # Get actual shield production from last_turns_shield_surplus
                shield_production = int(values[shield_surplus_idx])
                total_shields += shield_production

                # Estimate food and trade based on city size
                # Average city produces ~2 food per citizen
                total_food += city_size * 2.0
                # Average city produces ~1 trade per citizen
                total_trade += city_size * 1.0

            except (ValueError, IndexError):
                continue

        production[player_id] = {
            'food': total_food,
            'shields': total_shields,
            'trade': total_trade
        }

    return production


def parse_player_nations(savegame_content: str) -> Dict[int, int]:
    """Parse player nation IDs from savegame

    Args:
        savegame_content: Decompressed savegame file content

    Returns:
        Dict mapping player_id to nation_id
    """
    player_nations = {}

    # Find all player sections
    player_sections = re.finditer(r'\[player(\d+)\](.*?)\n\[', savegame_content, re.DOTALL)

    for player_match in player_sections:
        player_id = int(player_match.group(1))
        player_content = player_match.group(2)

        # Extract nation ID
        nation_match = re.search(r'nation="?([^"\n]+)"?', player_content)
        if nation_match:
            nation_str = nation_match.group(1)
            # Nation can be either a name (string) or ID (number)
            # For now, try to extract it as is - we'll handle lookup later
            player_nations[player_id] = nation_str

    return player_nations


def parse_player_science(savegame_content: str) -> Dict[int, Dict[str, any]]:
    """Parse science and technology data from savegame

    Args:
        savegame_content: Decompressed savegame file content

    Returns:
        Dict mapping player_id to science data:
        {
            player_id: {
                'science_per_turn': int,
                'techs_known': int,
                'researching': str,
                'research_bulbs': int
            }
        }
    """
    science_data = {}

    # Find all player sections
    player_sections = re.finditer(r'\[player(\d+)\](.*?)\n\[', savegame_content, re.DOTALL)

    for player_match in player_sections:
        player_id = int(player_match.group(1))
        player_content = player_match.group(2)

        # Extract science rate
        science_match = re.search(r'rates\.science=(\d+)', player_content)
        science_per_turn = int(science_match.group(1)) if science_match else 0

        # Extract bulbs from last turn
        bulbs_match = re.search(r'research\.bulbs_last_turn=(\d+)', player_content)
        bulbs_last_turn = int(bulbs_match.group(1)) if bulbs_match else 0

        science_data[player_id] = {
            'science_per_turn': science_per_turn,
            'research_bulbs': bulbs_last_turn,
            'techs_known': 0,  # Will be filled from research section
            'researching': None  # Will be filled from research section
        }

    # Parse research section for technology counts
    research_match = re.search(r'\[research\](.*?)\ncount=', savegame_content, re.DOTALL)
    if research_match:
        research_content = research_match.group(1)

        # Find research schema
        schema_match = re.search(r'r=\{([^}]+)\}', research_content)
        if schema_match:
            schema = schema_match.group(1).replace('"', '').split(',')

            try:
                number_idx = schema.index('number')
                techs_idx = schema.index('techs')
                now_name_idx = schema.index('now_name')
            except ValueError:
                return science_data

            # Find research data lines
            research_lines = re.findall(r'\n(\d+,[^\n]+)', research_content)

            for line in research_lines:
                # Parse CSV
                values = []
                current = ""
                in_quotes = False
                for char in line:
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        values.append(current)
                        current = ""
                    else:
                        current += char
                values.append(current)

                if len(values) > max(number_idx, techs_idx, now_name_idx):
                    try:
                        player_num = int(values[number_idx])
                        techs_count = int(values[techs_idx])
                        researching = values[now_name_idx].strip('"')

                        if player_num in science_data:
                            science_data[player_num]['techs_known'] = techs_count
                            science_data[player_num]['researching'] = researching if researching else None
                    except (ValueError, IndexError):
                        continue

    return science_data


def parse_player_technologies(savegame_content: str, ruleset_techs: Optional[Dict] = None) -> Dict[int, set]:
    """Parse detailed technology discoveries from savegame

    Args:
        savegame_content: Decompressed savegame file content
        ruleset_techs: Optional dict mapping tech_id to tech data from ruleset

    Returns:
        Dict mapping player_id to set of known tech IDs:
        {
            player_id: {tech_id1, tech_id2, ...}
        }
    """
    player_techs = {}

    # Find research section
    research_match = re.search(r'\[research\](.*?)\ncount=', savegame_content, re.DOTALL)
    if not research_match:
        return player_techs

    research_content = research_match.group(1)

    # Find research schema - note: it doesn't have closing }, just ends with newline
    schema_match = re.search(r'r=\{(.+)\n', research_content)
    if not schema_match:
        return player_techs

    schema = schema_match.group(1).replace('"', '').split(',')

    try:
        number_idx = schema.index('number')
        done_idx = schema.index('done')
    except ValueError:
        return player_techs

    # Find research data lines - they start with a number
    # Format: number,"goal_name",techs,futuretech,bulbs_before,"saved_name",bulbs,"now_name",free_bulbs,"done_binary_string"
    research_lines = re.findall(r'^(\d+,.+)$', research_content, re.MULTILINE)

    for line in research_lines:
        # Parse CSV carefully, handling quoted strings
        values = []
        current = ""
        in_quotes = False
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                values.append(current)
                current = ""
            else:
                current += char
        values.append(current)

        if len(values) <= max(number_idx, done_idx):
            continue

        try:
            player_id = int(values[number_idx])
            done_string = values[done_idx].strip('"')

            # Parse binary string - each '1' means tech is known
            # Tech IDs start at 0
            known_techs = set()
            for tech_id, bit in enumerate(done_string):
                if bit == '1':
                    known_techs.add(str(tech_id))

            player_techs[player_id] = known_techs

        except (ValueError, IndexError):
            continue

    return player_techs


def extract_complete_data_from_savegame(username: str, turn: int, host: str = 'localhost', port: int = 8080, recording_dir: Optional[str] = None) -> Optional[Dict[str, any]]:
    """Extract complete game data from savegame file

    Reads savegames from the local recordings directory. Savegames should be
    downloaded during the game recording phase (see run_world.py).

    Args:
        username: Player username
        turn: Turn number (will try turn and turn+1, as final save may be labeled as next turn)
        host: Server host (unused, kept for backward compatibility)
        port: Server port (unused, kept for backward compatibility)
        recording_dir: Path to recordings directory containing savegames/

    Returns:
        Dict with keys 'production', 'science', and 'nations' if successful, None otherwise:
        {
            'production': {player_id: {'food': float, 'shields': float, 'trade': float}},
            'science': {player_id: {'science_per_turn': int, 'techs_known': int, 'researching': str}},
            'nations': {player_id: nation_name_or_id}
        }
    """
    if not recording_dir:
        return None

    # Find savegame in local recordings directory
    savegame_name = find_local_savegame_for_turn(username, turn, recording_dir)
    if not savegame_name:
        return None

    # Load from local directory
    result = load_local_savegame(savegame_name, recording_dir)
    if not result:
        return None

    savegame_bytes, actual_filename = result

    try:
        # Decompress
        content = decompress_savegame_content(savegame_bytes, actual_filename)

        # Parse all data
        production = parse_city_production(content)
        science = parse_player_science(content)
        nations = parse_player_nations(content)
        technologies = parse_player_technologies(content)

        return {
            'production': production,
            'science': science,
            'nations': nations,
            'technologies': technologies
        }

    except Exception as e:
        print(f"Error parsing savegame {actual_filename}: {e}")
        import traceback
        traceback.print_exc()
        return None


# Keep old function for backward compatibility
def extract_production_from_savegame(username: str, turn: int, host: str = 'localhost', port: int = 8080) -> Optional[Dict[int, Dict[str, float]]]:
    """Extract complete production data from savegame (backward compatibility wrapper)"""
    result = extract_complete_data_from_savegame(username, turn, host, port)
    return result['production'] if result else None


def extract_username_from_config(config) -> str:
    """Extract username from report config

    Args:
        config: ReportConfig instance with recording_dir attribute

    Returns:
        Username string (defaults to 'myagent2' if not found)
    """
    username = 'myagent2'  # default
    if hasattr(config, 'recording_dir'):
        parts = config.recording_dir.rstrip('/').split('/')
        if 'recordings' in parts:
            idx = parts.index('recordings')
            if idx + 1 < len(parts):
                username = parts[idx + 1]
    return username


def get_savegame_data_for_report(config, turn: int) -> Optional[Dict[str, any]]:
    """Get complete savegame data for a specific turn in a report

    This is a convenience wrapper that extracts the username from config
    and retrieves savegame data for the specified turn. Automatically
    persists savegames to the recordings directory.

    Args:
        config: ReportConfig instance with recording_dir attribute
        turn: Turn number to get data for

    Returns:
        Dict with 'production' and 'science' keys if successful, None otherwise
    """
    username = extract_username_from_config(config)
    recording_dir = getattr(config, 'recording_dir', None)
    return extract_complete_data_from_savegame(username, turn, recording_dir=recording_dir)
