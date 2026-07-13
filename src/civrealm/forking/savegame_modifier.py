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
        # Newer save format: techs live in the [research] table, one row per
        # research number (== player id without team research), with a known-
        # techs count in column 3 and a "done" bitstring as the last field:
        #   0,"The Republic",8,0,2,"",30,"The Republic",0,"1010...01"
        research_idx = self.content.find('[research]')
        if research_idx != -1:
            row_pattern = re.compile(
                rf'^({player_id},"[^"]*",)(\d+)(,.*,")([01]+)("\s*)$', re.MULTILINE)

            def replace_row(match):
                head, count, mid, bits, tail = match.groups()
                if tech_id >= len(bits) or bits[tech_id] == '1':
                    return match.group(0)
                new_bits = bits[:tech_id] + '1' + bits[tech_id + 1:]
                return f'{head}{int(count) + 1}{mid}{new_bits}{tail}'

            section_end = self.content.find('\n[', research_idx + 1)
            section_end = len(self.content) if section_end == -1 else section_end
            section = self.content[research_idx:section_end]
            new_section, n = row_pattern.subn(replace_row, section)
            if n == 0:
                raise ValueError(
                    f"grant_player_tech: no [research] row matched for player {player_id}")
            self.content = (self.content[:research_idx] + new_section
                            + self.content[section_end:])
            return

        # Old save format: per-player inventions bitstring.
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

        new_content, n = re.subn(pattern, replace_tech, self.content, flags=re.DOTALL)
        if n == 0:
            raise ValueError(
                f"grant_player_tech: no inventions field matched for player {player_id}")
        self.content = new_content

    # -------------------------------------------------------------------------
    # RNG state mutation (for Monte Carlo rollouts)
    # -------------------------------------------------------------------------

    @staticmethod
    def _fc_srand_state(seed: int) -> tuple[list[int], int, int, int]:
        """Reproduce Freeciv's ``fc_srand(seed)`` in Python.

        Returns ``(v, j, k, x)`` matching the global ``rand_state`` Freeciv
        would hold after ``fc_srand(seed)``. Mirrors ``utility/rand.c``
        bit-for-bit: linear congruential init, then a 10000-iteration
        warm-up via the Mitchell-Moore additive generator with
        ``size = MAX_UINT32``.
        """
        MASK = 0xFFFFFFFF
        v = [0] * 56
        v[0] = seed & MASK
        for i in range(1, 56):
            v[i] = (3 * v[i - 1] + 257) & MASK
        j, k, x = 0, 31, 55

        # Heat-up: 10000 iterations of fc_rand(MAX_UINT32).
        # With size == MAX_UINT32, divisor == 1 and max == MAX_UINT32 - 1,
        # so the rejection branch only fires when v[j]+v[k] == MASK exactly.
        for _ in range(10000):
            while True:
                new_rand = (v[j] + v[k]) & MASK
                x = (x + 1) % 56
                j = (j + 1) % 56
                k = (k + 1) % 56
                v[x] = new_rand
                if new_rand <= MASK - 1:
                    break
        return v, j, k, x

    def set_rng_from_seed(self, seed: int) -> None:
        """Overwrite the savegame's ``[random]`` block with the state Freeciv
        would have after ``fc_srand(seed)``.

        Two Freeciv-only callers of ``fc_srand`` produce game RNG state:
        ``init_game_seed`` at game start, and (after loading) restoration of
        the saved table. By writing a ``fc_srand``-equivalent state directly
        into the savegame, we get a deterministic, seed-controlled rollout
        from this saved turn — without patching the server.
        """
        v, j, k, x = self._fc_srand_state(seed)

        def _table_line(words: list[int]) -> str:
            # Freeciv writes each word as %8x (lowercase, space-padded, no
            # leading zeros), space-separated, the whole thing in quotes.
            return '"' + ' '.join(f'{w:8x}' for w in words) + '"'

        new_block_lines = [
            '[random]',
            'saved=TRUE',
            f'index_J={j}',
            f'index_K={k}',
            f'index_X={x}',
        ]
        for t in range(8):
            words = v[t * 7:(t + 1) * 7]
            new_block_lines.append(f'table{t}={_table_line(words)}')
        new_block = '\n'.join(new_block_lines)

        # Replace the existing [random] block (up to the next [section] header).
        pattern = r'\[random\].*?(?=\n\[)'
        new_content, n = re.subn(pattern, new_block, self.content, count=1, flags=re.DOTALL)
        if n != 1:
            raise RuntimeError("Failed to locate [random] block in savegame")
        self.content = new_content

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
