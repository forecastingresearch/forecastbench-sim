#!/usr/bin/env python3
"""Run a Civilization game and collect savegames

This script runs an all-AI competitive game where ALL players are controlled
by Freeciv's built-in AI, then downloads the savegames for later processing.

Usage:
    python run_world.py --seed 42 --max_turns 50
    python run_world.py --seed 42 --quiet  # Suppress output for batch runs

The seed uniquely identifies the run and ensures deterministic gameplay.
Games with the same seed will produce identical results.

Setup:
- Creates N AI players (default 5) using aifill
- Connects as a player and toggles to Freeciv AI control via /aitoggle
- All players use the same Freeciv AI algorithm

Output:
- Recordings saved to: logs/recordings/seed{seed}/

Authentication note:
- The freeciv-proxy validates usernames via regex: [a-z][a-z0-9]* with 3-31 char length
- If usernames exist in the auth table with different passwords, authentication fails
- Solution: Use fresh usernames OR clear auth table: mysql -u docker -pchangeme freeciv_web -e 'DELETE FROM auth;'
"""

import sys
import argparse
import subprocess
import random
from pathlib import Path

# Add src to path for world report imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from civrealm.configs import fc_args
from civrealm.agents import NoOpAgent
from civrealm.world_reports.utils.savegame_parser import (
    download_all_savegames_from_docker
)
import gymnasium
import time

# Game Configuration defaults
AI_DIFFICULTY = 'hard'  # Options: 'handicapped', 'novice', 'easy', 'normal', 'hard', 'cheating', 'experimental'


def cleanup_docker_savegames(username: str, container_name: str = 'freeciv-web'):
    """Clean up any existing savegames for this username in Docker

    Args:
        username: Player username
        container_name: Docker container name
    """
    docker_path = f"/var/lib/tomcat10/webapps/data/savegames/{username}"

    # Remove existing directory
    subprocess.run(
        ['docker', 'exec', container_name, 'rm', '-rf', docker_path],
        capture_output=True
    )

    # Recreate empty directory
    subprocess.run(
        ['docker', 'exec', container_name, 'mkdir', '-p', docker_path],
        capture_output=True
    )


def main(seed: int, max_turns: int = 50, num_ai_players: int = 5, quiet: bool = False, load_game: str = ""):
    # Use seed as the unique identifier for this run
    # Username requirements (from freeciv-proxy validate_username):
    # - Must be 3-31 characters long
    # - Must start with a letter [a-z]
    # - Can only contain letters and numbers [a-z0-9]
    # - Cannot be "pbem"
    # Using 'seed' prefix ensures minimum 5 chars even for seed=0
    run_id = f'seed{seed}'
    fc_args['username'] = run_id
    fc_args['debug.record_action_and_observation'] = True
    fc_args['max_turns'] = max_turns
    fc_args['aifill'] = num_ai_players
    fc_args['begin_turn_timeout'] = 120  # Longer timeout for stable long games

    # Set seed for deterministic runs
    fc_args['debug.randomly_generate_seeds'] = False
    fc_args['debug.mapseed'] = seed
    fc_args['debug.gameseed'] = seed
    # Also seed Python's random module for nation selection (used in civ_controller.py)
    random.seed(seed)

    # Load game support - continue from a previous savegame
    if load_game:
        fc_args['debug.load_game'] = load_game
        fc_args['debug.take_player'] = run_id  # Take control of our player
        fc_args['begin_turn_timeout'] = 60  # Increase timeout for loaded games

    def log(msg):
        if not quiet:
            print(msg)

    if load_game:
        log(f"Continuing game from savegame: {load_game}")
    else:
        log("Starting all-AI game collection...")
    log(f"Seed: {seed}")
    log(f"AI Players: {num_ai_players} total (all Freeciv AI at {AI_DIFFICULTY} difficulty)")
    if not load_game:
        log(f"Setup: {num_ai_players - 1} via aifill + 1 connected player toggled to AI")
    log(f"Max turns: {max_turns} (fc_args: {fc_args['max_turns']})")
    log(f"Recording to: logs/recordings/{run_id}/")
    log("")

    # Clean up any existing savegames for this run (skip if loading)
    if not load_game:
        cleanup_docker_savegames(run_id)

    # Disable env checker when loading games (observations may be empty initially)
    if load_game:
        env = gymnasium.make('civrealm/FreecivBase-v0', disable_env_checker=True)
    else:
        env = gymnasium.make('civrealm/FreecivBase-v0')
    # NoOpAgent just ends turn - connected player will be toggled to Freeciv AI
    agent = NoOpAgent()

    observations, info = env.reset()

    # Note: Fog of war is disabled in client_state.py set_multiplayer_game()
    # This ensures complete world data for reports

    # Preserve all autosaves throughout the game for complete data extraction
    env.unwrapped.civ_controller.delete_save = False

    # Note: DO NOT enter observer mode - it prevents autosaves on turns 2-50
    # Observer mode causes handle_begin_turn to exit early without calling save_game()

    # AI difficulty is set to hard in client_state.py set_multiplayer_game()
    # via /set skilllevel hard (before aifill) and /hard (after aifill)

    # NOTE: phasemode=PLAYER doesn't work with singleplayer + NoOpAgent setup
    # It causes the game to hang waiting for explicit turn control
    # In singleplayer mode, Freeciv uses concurrent turns with built-in randomization
    # which helps mitigate first-mover advantage automatically

    # Skip initial setup if loading a game (settings already in savegame)
    if not load_game:
        # Randomize starting position assignments to balance the game
        # teamplacement=DISABLED assigns starting positions randomly rather than by team
        log("Randomizing starting positions...")
        env.unwrapped.civ_controller.ws_client.send_message("/set teamplacement DISABLED")
        time.sleep(0.5)

        # Toggle the connected player to be AI-controlled by Freeciv's built-in AI
        log(f"Toggling {fc_args['username']} to Freeciv AI control...")
        env.unwrapped.civ_controller.ws_client.send_message(f"/aitoggle {fc_args['username']}")
        time.sleep(0.5)

        # Set the connected player to hard difficulty (after aitoggle makes it an AI)
        log(f"Setting {fc_args['username']} to hard difficulty...")
        env.unwrapped.civ_controller.ws_client.send_message(f"/hard {fc_args['username']}")
        time.sleep(0.5)

        # Aifill players are already AI-controlled by default (PLRF_AI flag set)
        # DO NOT toggle them - that would turn OFF their AI!
        log(f"All {num_ai_players - 1} aifill players are AI-controlled by default")
    else:
        # When loading, toggle player back to AI control
        log(f"Toggling {fc_args['username']} back to Freeciv AI control...")
        env.unwrapped.civ_controller.ws_client.send_message(f"/aitoggle {fc_args['username']}")
        time.sleep(0.5)

    done = False
    step = 0

    log(f"Game started - all {num_ai_players} players controlled by Freeciv AI")
    log(f"Running for up to {max_turns} turns (AI vs AI competitive game)")
    log("")

    while not done:
        try:
            # NoOpAgent returns None, ending turn and letting Freeciv AI play
            action = agent.act(observations, info)
            observations, reward, terminated, truncated, info = env.step(action)

            turn = info.get('turn', 0)
            if turn > 0 and turn % 10 == 0:
                log(f"Turn {turn}/{max_turns}")

            step += 1
            # Environment will set terminated=True when max_turns is reached
            done = terminated or truncated

        except Exception as e:
            if not quiet:
                print(f"Error: {e}")
            raise e

    # Save and preserve the final game state for extracting complete production data
    # Autosave only happens at the beginning of turns, so we need to manually save at the end
    env.unwrapped.civ_controller.save_game()
    env.unwrapped.civ_controller.delete_save = False  # Prevent deletion

    env.close()

    # Download and persist all savegames from Docker container
    recording_dir = Path(__file__).parent.parent / 'logs' / 'recordings' / run_id
    recording_dir.mkdir(parents=True, exist_ok=True)
    downloaded, skipped, failed = download_all_savegames_from_docker(run_id, str(recording_dir))
    log(f"Downloaded {downloaded} savegames (skipped {skipped} existing, {failed} failed)")

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run an all-AI Civilization game and collect savegames'
    )
    parser.add_argument(
        '--seed',
        type=int,
        required=True,
        help='Random seed for the game (used as unique run identifier and for deterministic gameplay)'
    )
    parser.add_argument(
        '--max_turns',
        type=int,
        default=50,
        help='Maximum number of turns to run (default: 50)'
    )
    parser.add_argument(
        '--num_ai_players',
        type=int,
        default=5,
        help='Total number of AI players in the game (default: 5)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output (useful for batch runs)'
    )
    parser.add_argument(
        '--load_game',
        type=str,
        default="",
        help='Load and continue from a savegame (e.g., seed0_T260_2026-01-30-18_01)'
    )

    args = parser.parse_args()
    exit(main(
        seed=args.seed,
        max_turns=args.max_turns,
        num_ai_players=args.num_ai_players,
        quiet=args.quiet,
        load_game=args.load_game
    ))
