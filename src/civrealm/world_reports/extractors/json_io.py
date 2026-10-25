"""JSON I/O utilities for world report data"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def write_world_data(data: Dict[str, Any], filepath: str) -> None:
    """Write world report data to JSON file

    Args:
        data: World report data dictionary
        filepath: Path to output JSON file
    """
    # Ensure output directory exists
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add generation timestamp if not present
    if 'metadata' in data and 'generated_at' not in data['metadata']:
        data['metadata']['generated_at'] = datetime.now().isoformat()

    # Write JSON with nice formatting
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Print file size for feedback
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    Written: {filepath} ({size_mb:.2f} MB)")


def read_world_data(filepath: str) -> Dict[str, Any]:
    """Read world report data from JSON file

    Args:
        filepath: Path to input JSON file

    Returns:
        World report data dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Basic validation
    validate_schema(data)

    return data


def validate_schema(data: Dict[str, Any]) -> None:
    """Validate that world report data has required structure

    Args:
        data: World report data dictionary

    Raises:
        ValueError: If required fields are missing
    """
    required_top_level = ['metadata', 'civilizations', 'time_series', 'events', 'snapshots']

    for field in required_top_level:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Validate metadata fields
    required_metadata = ['turn', 'turns_analyzed']
    for field in required_metadata:
        if field not in data['metadata']:
            raise ValueError(f"Missing required metadata field: {field}")

    # Validate civilizations is a dict
    if not isinstance(data['civilizations'], dict):
        raise ValueError("'civilizations' must be a dictionary")

    # Validate time_series is a dict
    if not isinstance(data['time_series'], dict):
        raise ValueError("'time_series' must be a dictionary")

    # Validate events is a list
    if not isinstance(data['events'], list):
        raise ValueError("'events' must be a list")

    # Validate snapshots is a dict
    if not isinstance(data['snapshots'], dict):
        raise ValueError("'snapshots' must be a dictionary")
