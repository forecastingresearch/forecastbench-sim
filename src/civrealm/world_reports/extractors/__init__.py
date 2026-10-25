"""Data extraction layer for world reports

This module handles extracting processed game data from state files
and savegames into a structured format for rendering.
"""

from .metrics_collector import MetricsCollector
from .json_io import write_world_data, read_world_data, validate_schema

__all__ = [
    'MetricsCollector',
    'write_world_data',
    'read_world_data',
    'validate_schema',
]
