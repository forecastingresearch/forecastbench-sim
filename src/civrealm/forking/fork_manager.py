"""Fork Manager for running parallel game simulations from checkpoints.

Orchestrates running multiple game forks from a checkpoint with different
event injections, enabling conditional forecasting (P(B|A) questions).

IMPORTANT: Global State Issue
-----------------------------
The civrealm library has global singleton state (Ports, fc_args) that prevents
running multiple forks sequentially in the same Python process. The first fork
will succeed, but subsequent forks will fail with timeout errors.

**Solution:** Run each fork in a separate subprocess. See scripts/test_determinism.py
for the recommended pattern using subprocess isolation.

This limitation affects:
- run_fork(): Only reliable for ONE fork per process
- run_all_forks(): Will fail on the second fork

The subprocess pattern:
    import subprocess, json, sys

    script = '''
    from civrealm.forking import ForkManager
    manager = ForkManager("logs/recordings/s100", 100)
    fork = manager.create_fork(checkpoint_turn=50, modifications=[], fork_name="test")
    result = manager.run_fork(fork, 60)
    print("RESULT:" + json.dumps({"success": result.success, ...}))
    '''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True)

Example usage (single fork per process):
    manager = ForkManager("logs/recordings/s100", base_seed=100)

    # Create forks with different modifications
    baseline = manager.create_fork(
        checkpoint_turn=50,
        modifications=[],
        fork_name="baseline"
    )

    gold_boost = manager.create_fork(
        checkpoint_turn=50,
        modifications=[{"type": "gold", "player_id": 0, "value": 5000}],
        fork_name="gold5000"
    )

    # Run ONE fork (use subprocess for multiple forks)
    result = manager.run_fork(baseline, end_turn=100)
"""

import shutil
import subprocess
import tempfile
import time
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium

from civrealm.agents import NoOpAgent
from civrealm.configs import fc_args
from civrealm.forking.savegame_modifier import SavegameModifier
from civrealm.freeciv.utils.port_utils import Ports
from civrealm.world_reports.utils.savegame_parser import (
    download_all_savegames_from_docker,
    decompress_savegame_content,
)


@dataclass
class Fork:
    """Represents a single forked game simulation."""

    name: str
    base_savegame_path: str  # Local path to .sav.xz
    modifications: List[dict]  # [{type: "gold", player_id: 0, value: 1000}]
    output_dir: str
    username: str  # For Docker paths and game identification


@dataclass
class ForkResult:
    """Results from running a fork."""

    fork_name: str
    final_turn: int
    player_states: Dict[int, dict]  # Player metrics at end
    success: bool
    error: Optional[str] = None


class ForkManager:
    """Orchestrates running multiple game forks from checkpoints."""

    def __init__(self, base_recording_dir: str, base_seed: int):
        """Initialize the fork manager.

        Args:
            base_recording_dir: Path to recording directory containing savegames/
                               (e.g., "logs/recordings/s100")
            base_seed: The seed of the base game (used for fork naming)
        """
        self.base_dir = Path(base_recording_dir)
        self.base_seed = base_seed
        self.forks: List[Fork] = []
        self.results: Dict[str, ForkResult] = {}

        # Verify savegames directory exists
        self.savegames_dir = self.base_dir / "savegames"
        if not self.savegames_dir.exists():
            raise ValueError(f"Savegames directory not found: {self.savegames_dir}")

    def find_savegame(self, turn: int) -> Optional[Path]:
        """Find the savegame file closest to the requested turn.

        Args:
            turn: Target turn number

        Returns:
            Path to savegame file, or None if not found
        """
        # List all savegames
        savegames = list(self.savegames_dir.glob("*.sav.*"))

        if not savegames:
            return None

        # Parse turn numbers from filenames
        # Format: s{seed}_T{turn}_{timestamp}.sav.xz or .sav.zst
        turn_files = []
        for f in savegames:
            try:
                # Extract turn from filename: username_T{turn}_...
                parts = f.stem.replace(".sav", "").split("_")
                for i, part in enumerate(parts):
                    if part.startswith("T") and part[1:].isdigit():
                        file_turn = int(part[1:])
                        turn_files.append((file_turn, f))
                        break
            except (ValueError, IndexError):
                continue

        if not turn_files:
            return None

        # Find exact match or closest turn <= requested turn
        exact = [f for t, f in turn_files if t == turn]
        if exact:
            return exact[0]

        # Get closest turn that doesn't exceed requested turn
        candidates = [(t, f) for t, f in turn_files if t <= turn]
        if candidates:
            return max(candidates, key=lambda x: x[0])[1]

        # If all turns are greater, return the smallest
        return min(turn_files, key=lambda x: x[0])[1]

    def create_fork(
        self,
        checkpoint_turn: int,
        modifications: List[dict],
        fork_name: str
    ) -> Fork:
        """Create a fork definition from a checkpoint.

        Args:
            checkpoint_turn: Turn to fork from
            modifications: List of modifications to apply
                          Each modification is a dict with 'type' and type-specific params:
                          - {"type": "gold", "player_id": 0, "value": 5000}
                          - {"type": "government", "player_id": 0, "value": "Republic"}
                          - {"type": "tech", "player_id": 0, "tech_id": 23}
            fork_name: Unique name for this fork

        Returns:
            Fork object
        """
        # Find the savegame for this turn
        savegame_path = self.find_savegame(checkpoint_turn)
        if savegame_path is None:
            raise ValueError(f"No savegame found for turn {checkpoint_turn}")

        # Create unique username for this fork
        # Note: username CANNOT contain underscores - Freeciv extracts host_name
        # from savegame filename by splitting on '_', so underscores in username
        # will break the load_game authentication.
        username = f"s{self.base_seed}fork{fork_name}"

        # Create output directory for this fork
        output_dir = str(self.base_dir.parent / username)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        fork = Fork(
            name=fork_name,
            base_savegame_path=str(savegame_path),
            modifications=modifications,
            output_dir=output_dir,
            username=username
        )

        self.forks.append(fork)
        return fork

    def _apply_modifications(
        self,
        savegame_path: str,
        modifications: List[dict],
        output_path: str
    ) -> str:
        """Apply modifications to a savegame and save to new location.

        Args:
            savegame_path: Path to original savegame
            modifications: List of modifications to apply
            output_path: Where to save the modified savegame

        Returns:
            Path to modified savegame
        """
        modifier = SavegameModifier(savegame_path)

        for mod in modifications:
            mod_type = mod.get("type")

            if mod_type == "gold":
                modifier.set_player_gold(mod["player_id"], mod["value"])
            elif mod_type == "government":
                modifier.set_player_government(mod["player_id"], mod["value"])
            elif mod_type == "tech":
                modifier.grant_player_tech(mod["player_id"], mod["tech_id"])
            else:
                raise ValueError(f"Unknown modification type: {mod_type}")

        # Save to output path
        modifier.save(output_path)
        return output_path

    def _cleanup_docker_savegames(
        self,
        username: str,
        container_name: str = "freeciv-web"
    ):
        """Clean up any existing savegames for this username in Docker.

        Args:
            username: Player username
            container_name: Docker container name
        """
        docker_path = f"/var/lib/tomcat10/webapps/data/savegames/{username}"

        # Remove existing directory
        subprocess.run(
            ["docker", "exec", container_name, "rm", "-rf", docker_path],
            capture_output=True
        )

        # Recreate empty directory
        subprocess.run(
            ["docker", "exec", container_name, "mkdir", "-p", docker_path],
            capture_output=True
        )

    def _upload_savegame_to_docker(
        self,
        local_path: str,
        username: str,
        container_name: str = "freeciv-web"
    ) -> str:
        """Upload a savegame to the Docker container.

        Args:
            local_path: Local path to the savegame file
            username: Username for the Docker path
            container_name: Docker container name

        Returns:
            The savegame name for use with /load command
        """
        local_file = Path(local_path)
        savegame_name = local_file.name
        docker_path = f"/var/lib/tomcat10/webapps/data/savegames/{username}/{savegame_name}"

        # Copy to container
        subprocess.run(
            ["docker", "cp", local_path, f"{container_name}:{docker_path}"],
            check=True
        )

        # Return the savegame name without extension for /load command
        return savegame_name.replace(".sav.xz", "").replace(".sav.zst", "")

    def _extract_player_states(self, env) -> Dict[int, dict]:
        """Extract current player states from the environment.

        Args:
            env: The FreeCiv environment

        Returns:
            Dict mapping player_id to player metrics
        """
        player_states = {}

        controller = env.unwrapped.civ_controller
        player_ctrl = controller.player_ctrl

        for player_id, player in player_ctrl.players.items():
            player_states[player_id] = {
                "name": player.get("name", "Unknown"),
                "nation": player.get("nation", -1),
                "score": player.get("score", 0),
                "gold": player.get("gold", 0),
                "is_alive": player.get("is_alive", False),
                "researching": player.get("researching", None),
            }

        return player_states

    def run_fork(self, fork: Fork, end_turn: int) -> ForkResult:
        """Run a single fork from checkpoint to end_turn.

        Args:
            fork: Fork definition to run
            end_turn: Turn to stop at

        Returns:
            ForkResult with outcomes
        """
        print(f"Running fork '{fork.name}'...")

        try:
            # Create modified savegame in temp location
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy original savegame to temp
                original_name = Path(fork.base_savegame_path).name
                # Rename to use fork's username
                modified_name = original_name.replace(
                    original_name.split("_")[0],  # Replace original username prefix
                    fork.username
                )
                temp_savegame = Path(temp_dir) / modified_name

                # Apply modifications (always rename player0 to match fork username)
                modifier = SavegameModifier(fork.base_savegame_path)
                modifier.set_player_name(0, fork.username)
                for mod in fork.modifications:
                    mod_type = mod.get("type")
                    if mod_type == "gold":
                        modifier.set_player_gold(mod["player_id"], mod["value"])
                    elif mod_type == "government":
                        modifier.set_player_government(mod["player_id"], mod["value"])
                    elif mod_type == "tech":
                        modifier.grant_player_tech(mod["player_id"], mod["tech_id"])
                modifier.save(str(temp_savegame))

                # Clean up Docker and upload
                self._cleanup_docker_savegames(fork.username)
                savegame_load_name = self._upload_savegame_to_docker(
                    str(temp_savegame),
                    fork.username
                )

                # Configure fc_args for this fork - reset all relevant state
                fc_args["username"] = fork.username
                fc_args["debug.load_game"] = savegame_load_name
                fc_args["debug.take_player"] = fork.username  # Take control of our player
                fc_args["debug.record_action_and_observation"] = True
                fc_args["max_turns"] = end_turn
                fc_args["begin_turn_timeout"] = 60  # Increase timeout for loaded games

                # Seed random for consistency
                random.seed(self.base_seed)

                # Get a port for this fork - clear all Ports state first
                Ports.clear()
                Ports._cache = {}
                port = Ports.get()
                fc_args["client_port"] = port
                print(f"Reset with port: {port}")

                # Create environment and run
                env = gymnasium.make("civrealm/FreecivBase-v0")
                agent = NoOpAgent()

                observations, info = env.reset()

                # Preserve all autosaves
                env.unwrapped.civ_controller.delete_save = False

                # Toggle player to AI control
                env.unwrapped.civ_controller.ws_client.send_message(
                    f"/aitoggle {fork.username}"
                )
                time.sleep(0.5)

                # Run game loop
                done = False
                final_turn = info.get("turn", 0)

                while not done:
                    action = agent.act(observations, info)
                    observations, reward, terminated, truncated, info = env.step(action)
                    final_turn = info.get("turn", final_turn)
                    done = terminated or truncated

                # Extract final player states
                player_states = self._extract_player_states(env)

                # Save final game state
                env.unwrapped.civ_controller.save_game()
                env.unwrapped.civ_controller.delete_save = False

                env.close()

                # Wait for port to be released and server state to clear
                time.sleep(5)

                # Download savegames to fork output directory
                download_all_savegames_from_docker(
                    fork.username,
                    fork.output_dir
                )

                result = ForkResult(
                    fork_name=fork.name,
                    final_turn=final_turn,
                    player_states=player_states,
                    success=True
                )
                self.results[fork.name] = result
                return result

        except Exception as e:
            result = ForkResult(
                fork_name=fork.name,
                final_turn=0,
                player_states={},
                success=False,
                error=str(e)
            )
            self.results[fork.name] = result
            return result

    def run_all_forks(self, end_turn: int) -> Dict[str, ForkResult]:
        """Run all registered forks sequentially.

        WARNING: Due to global state in civrealm (Ports singleton, fc_args),
        only the FIRST fork will succeed. Subsequent forks will fail with
        timeout errors. Use subprocess isolation for multiple forks - see
        scripts/test_determinism.py for the recommended pattern.

        Args:
            end_turn: Turn to stop at for all forks

        Returns:
            Dict mapping fork name to ForkResult
        """
        if len(self.forks) > 1:
            print("WARNING: Running multiple forks in same process - only first will succeed.")
            print("         Use subprocess isolation for reliable multi-fork execution.")

        results = {}

        for fork in self.forks:
            result = self.run_fork(fork, end_turn)
            results[fork.name] = result
            print(f"Fork '{fork.name}': {'SUCCESS' if result.success else 'FAILED'}")
            if result.error:
                print(f"  Error: {result.error}")

        return results

    def compare_outcomes(self, metric: str) -> Dict[str, Any]:
        """Compare a specific metric across all fork results.

        Args:
            metric: Metric to compare. Options:
                   - "player_gold": Gold for each player
                   - "player_score": Score for each player
                   - "final_turn": Final turn reached

        Returns:
            Dict mapping fork name to metric value
        """
        comparison = {}

        for fork_name, result in self.results.items():
            if not result.success:
                comparison[fork_name] = None
                continue

            if metric == "final_turn":
                comparison[fork_name] = result.final_turn
            elif metric == "player_gold":
                comparison[fork_name] = {
                    pid: state.get("gold", 0)
                    for pid, state in result.player_states.items()
                }
            elif metric == "player_score":
                comparison[fork_name] = {
                    pid: state.get("score", 0)
                    for pid, state in result.player_states.items()
                }
            elif metric.startswith("player_"):
                # Generic player metric extraction
                field = metric.replace("player_", "")
                comparison[fork_name] = {
                    pid: state.get(field)
                    for pid, state in result.player_states.items()
                }
            else:
                comparison[fork_name] = None

        return comparison
