"""
Savegame modification for world forking experiments.

Modifies FreeCiv savegame files to inject events/changes before reloading.
This enables conditional forecasting: "P(B | A happened)" with ground-truth from simulation.
"""

import lzma
import re
import tempfile
import subprocess
from pathlib import Path
from typing import Optional


class SavegameModifier:
    """Modify FreeCiv savegame files to inject events."""

    def __init__(self, savegame_path: str):
        """
        Initialize with path to a compressed savegame (.sav.xz).

        Args:
            savegame_path: Path to the .sav.xz file
        """
        self.savegame_path = Path(savegame_path)
        self.content: Optional[str] = None
        self._load()

    def _load(self) -> None:
        """Load and decompress the savegame."""
        with lzma.open(self.savegame_path, 'rt') as f:
            self.content = f.read()

    def save(self, output_path: Optional[str] = None) -> str:
        """
        Save the modified savegame.

        Args:
            output_path: Where to save. If None, overwrites original.

        Returns:
            Path to saved file.
        """
        if output_path is None:
            output_path = str(self.savegame_path)

        output_path = Path(output_path)
        with lzma.open(output_path, 'wt') as f:
            f.write(self.content)

        return str(output_path)

    def save_uncompressed(self, output_path: str) -> str:
        """Save without compression (for debugging)."""
        with open(output_path, 'w') as f:
            f.write(self.content)
        return output_path

    # -------------------------------------------------------------------------
    # Player modifications
    # -------------------------------------------------------------------------

    def set_player_name(self, player_id: int, name: str) -> None:
        """
        Set a player's name.

        Args:
            player_id: Player number (0-indexed)
            name: New player name
        """
        pattern = rf'(\[player{player_id}\].*?name=")[^"]*"'
        replacement = rf'\g<1>{name}"'
        self.content = re.sub(pattern, replacement, self.content, flags=re.DOTALL)

    def set_player_gold(self, player_id: int, gold: int) -> None:
        """
        Set a player's gold amount.

        Args:
            player_id: Player number (0-indexed)
            gold: New gold amount
        """
        pattern = rf'(\[player{player_id}\].*?gold=)\d+'
        replacement = rf'\g<1>{gold}'
        self.content = re.sub(pattern, replacement, self.content, flags=re.DOTALL)

    def add_player_gold(self, player_id: int, amount: int) -> None:
        """
        Add gold to a player's current amount.

        Args:
            player_id: Player number (0-indexed)
            amount: Gold amount to add (can be negative)
        """
        current = self.get_player_gold(player_id)
        self.set_player_gold(player_id, current + amount)

    def set_player_government(self, player_id: int, government: str) -> None:
        """
        Set a player's government type.

        Args:
            player_id: Player number
            government: Government name (e.g., "Republic", "Monarchy", "Democracy")
        """
        pattern = rf'(\[player{player_id}\].*?government_name=")[^"]*"'
        replacement = rf'\g<1>{government}"'
        self.content = re.sub(pattern, replacement, self.content, flags=re.DOTALL)

    def grant_player_tech(self, player_id: int, tech_id: int) -> None:
        """
        Grant a technology to a player.

        Technologies are stored as a binary string where each position
        represents whether that tech is known (1) or not (0).

        Args:
            player_id: Player number
            tech_id: Technology ID to grant
        """
        # Find the player's tech section
        pattern = rf'(\[player{player_id}\].*?research="inventions",")[01]*"'

        def replace_tech(match):
            prefix = match.group(1)
            # Extract the current tech string
            full_match = match.group(0)
            tech_string = full_match.split('"')[-2]

            # Convert to list, modify, convert back
            tech_list = list(tech_string)
            if tech_id < len(tech_list):
                tech_list[tech_id] = '1'
            tech_string = ''.join(tech_list)

            return f'{prefix}{tech_string}"'

        self.content = re.sub(pattern, replace_tech, self.content, flags=re.DOTALL)

    # -------------------------------------------------------------------------
    # Game state queries
    # -------------------------------------------------------------------------

    def get_player_gold(self, player_id: int) -> int:
        """Get current gold for a player."""
        pattern = rf'\[player{player_id}\].*?gold=(\d+)'
        match = re.search(pattern, self.content, flags=re.DOTALL)
        if match:
            return int(match.group(1))
        return 0

    def get_turn(self) -> int:
        """Get the current turn number."""
        match = re.search(r'turn=(\d+)', self.content)
        if match:
            return int(match.group(1))
        return 0

    def get_player_count(self) -> int:
        """Count number of players in the savegame."""
        return len(re.findall(r'\[player\d+\]', self.content))

    # -------------------------------------------------------------------------
    # Docker integration
    # -------------------------------------------------------------------------

    def upload_to_docker(self, username: str, container_name: str = 'freeciv-web') -> str:
        """
        Upload modified savegame to Docker container.

        Args:
            username: FreeCiv username (determines save directory)
            container_name: Docker container name

        Returns:
            The savegame name (without path) for use with /load command
        """
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.sav.xz', delete=False) as f:
            temp_path = f.name

        self.save(temp_path)

        # Determine docker path
        savegame_name = self.savegame_path.name
        docker_path = f'/var/lib/tomcat10/webapps/data/savegames/{username}/{savegame_name}'

        # Copy to container
        subprocess.run(
            ['docker', 'cp', temp_path, f'{container_name}:{docker_path}'],
            check=True
        )

        # Clean up temp file
        Path(temp_path).unlink()

        # Return the savegame name (without extension) for /load command
        return savegame_name.replace('.sav.xz', '')
