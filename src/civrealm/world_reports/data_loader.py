"""Load and index game state recordings"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class DataLoader:
    """Load and provide access to game state recordings

    Handles loading state JSON files from the recordings directory and
    provides efficient access to game states by turn number.
    """

    def __init__(self, recording_dir: str):
        """Initialize DataLoader

        Args:
            recording_dir: Path to logs/recordings/username/ directory
        """
        self.recording_dir = Path(recording_dir)
        if not self.recording_dir.exists():
            raise FileNotFoundError(f"Recording directory not found: {recording_dir}")

        # Index: turn -> list of (step, filepath)
        self._state_index: Dict[int, List[Tuple[int, Path]]] = defaultdict(list)
        self._build_index()

        # Load ruleset data (nations, etc.)
        self.ruleset = self._load_ruleset()

    def _build_index(self):
        """Build index of all state files by turn and step"""
        pattern = re.compile(r'turn_(\d+)_step_(\d+)_state\.json')

        for filepath in self.recording_dir.glob('turn_*_step_*_state.json'):
            match = pattern.match(filepath.name)
            if match:
                turn = int(match.group(1))
                step = int(match.group(2))
                self._state_index[turn].append((step, filepath))

        # Sort steps within each turn
        for turn in self._state_index:
            self._state_index[turn].sort(key=lambda x: x[0])

    def _load_ruleset(self) -> Optional[Dict]:
        """Load ruleset data (nations, etc.) from recording directory

        Returns:
            Dict containing ruleset data, or None if not found
        """
        ruleset_file = self.recording_dir / 'ruleset.json'
        if ruleset_file.exists():
            return self._load_json(ruleset_file)
        else:
            print(f"Warning: No ruleset.json found in {self.recording_dir}")
            print("Civilization names will not be available in reports.")
            return None

    def get_available_turns(self) -> List[int]:
        """Get list of all turns that have recorded states

        Returns:
            Sorted list of turn numbers
        """
        return sorted(self._state_index.keys())

    def get_max_turn(self) -> int:
        """Get the highest turn number available

        Returns:
            Maximum turn number
        """
        turns = self.get_available_turns()
        return max(turns) if turns else 0

    def get_state(self, turn: int, step: Optional[int] = None) -> Optional[Dict]:
        """Load game state for a specific turn

        Args:
            turn: Turn number
            step: Specific step within turn (if None, returns first step)

        Returns:
            Dict containing game state, or None if not found
        """
        if turn not in self._state_index:
            return None

        steps = self._state_index[turn]
        if not steps:
            return None

        # Find the requested step or use first step if not specified
        if step is None:
            filepath = steps[0][1]
        else:
            matching = [fp for s, fp in steps if s == step]
            if not matching:
                return None
            filepath = matching[0]

        return self._load_json(filepath)

    def get_states_range(self, start_turn: int, end_turn: int,
                        step: Optional[int] = None) -> Dict[int, Dict]:
        """Load states for a range of turns

        Args:
            start_turn: Starting turn (inclusive)
            end_turn: Ending turn (inclusive)
            step: Specific step within each turn (if None, uses first step)

        Returns:
            Dict mapping turn number to state dict
        """
        states = {}
        for turn in range(start_turn, end_turn + 1):
            state = self.get_state(turn, step)
            if state is not None:
                states[turn] = state
        return states

    def get_all_states_for_turn(self, turn: int) -> Dict[int, Dict]:
        """Get all steps for a specific turn

        Args:
            turn: Turn number

        Returns:
            Dict mapping step number to state dict
        """
        if turn not in self._state_index:
            return {}

        states = {}
        for step, filepath in self._state_index[turn]:
            state = self._load_json(filepath)
            if state is not None:
                states[step] = state
        return states

    def _load_json(self, filepath: Path) -> Optional[Dict]:
        """Load and parse a JSON file

        Args:
            filepath: Path to JSON file

        Returns:
            Parsed JSON as dict, or None on error
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load {filepath}: {e}")
            return None

    def get_turn_summary(self) -> Dict:
        """Get summary of available data

        Returns:
            Dict with summary statistics
        """
        turns = self.get_available_turns()
        total_files = sum(len(steps) for steps in self._state_index.values())

        return {
            'recording_dir': str(self.recording_dir),
            'total_turns': len(turns),
            'total_files': total_files,
            'turn_range': (min(turns), max(turns)) if turns else (0, 0),
            'turns_available': turns
        }
