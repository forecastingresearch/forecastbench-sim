#!/usr/bin/env python3
"""Run a Civilization game and generate world reports

This script runs an all-AI competitive game where ALL players are controlled
by Freeciv's built-in AI, then automatically generates world reports from the
recorded gameplay.

Setup:
- Sets aifill=5 to create 5 total players in the game
- Connects as player 'myagent2' (1 of the 5)
- Toggles myagent2 to Freeciv AI control via /aitoggle
- The other 4 players are AI-controlled by default
- Result: 5 Freeciv AI players, ALL using the same strategy

NoOpAgent's role:
- Simply returns None to end turn immediately
- Freeciv AI actually plays for this player
- This allows CivRealm to maintain connection and record observations

Result: A competitive 5-player all-AI game with full recording coverage.
All players use the same Freeciv AI algorithm, ensuring fair comparison.

After the game completes, world reports are automatically generated.
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add src to path for world report imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from civrealm.configs import fc_args
from civrealm.agents import NoOpAgent
from civrealm.world_reports import ReportGenerator, ReportConfig
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

    print(f"Cleaned Docker savegames directory for {username}")


def main(session_id: int = 0, max_turns: int = 50, num_ai_players: int = 5):
    # Configure username with session ID
    username = f'myagent{session_id}'
    fc_args['username'] = username
    fc_args['debug.record_action_and_observation'] = True
    fc_args['max_turns'] = max_turns
    fc_args['aifill'] = num_ai_players

    print("Starting all-AI game collection...")
    print(f"Session ID: {session_id}")
    print(f"Username: {username} (will be toggled to AI control)")
    print(f"AI Players: {num_ai_players} total (all Freeciv AI at {AI_DIFFICULTY} difficulty)")
    print(f"Setup: {num_ai_players - 1} via aifill + 1 connected player toggled to AI")
    print(f"Max turns: {max_turns}")
    print(f"Recording to: logs/recordings/{username}/")
    print()

    # Clean up any existing savegames for this username
    print("Cleaning Docker savegames directory...")
    cleanup_docker_savegames(username)
    print()

    env = gymnasium.make('civrealm/FreecivBase-v0')
    # NoOpAgent just ends turn - connected player will be toggled to Freeciv AI
    agent = NoOpAgent()

    observations, info = env.reset()

    # Note: Fog of war is disabled in client_state.py set_multiplayer_game()
    # This ensures complete world data for reports

    # Preserve all autosaves throughout the game for complete data extraction
    print(f"Preserving autosaves for complete historical data...")
    env.unwrapped.civ_controller.delete_save = False

    # Note: DO NOT enter observer mode - it prevents autosaves on turns 2-50
    # Observer mode causes handle_begin_turn to exit early without calling save_game()
    # print(f"Entering observer mode for complete data access...")
    # env.unwrapped.civ_controller.ws_client.send_message("/observe")
    # time.sleep(1)

    # Set AI difficulty level for all AI players
    print(f"Setting AI difficulty to {AI_DIFFICULTY}...")
    env.unwrapped.civ_controller.ws_client.send_message(f"/set skilllevel {AI_DIFFICULTY}")
    time.sleep(1)

    # NOTE: phasemode=PLAYER doesn't work with singleplayer + NoOpAgent setup
    # It causes the game to hang waiting for explicit turn control
    # In singleplayer mode, Freeciv uses concurrent turns with built-in randomization
    # which helps mitigate first-mover advantage automatically

    # Randomize starting position assignments to balance the game
    # teamplacement=DISABLED assigns starting positions randomly rather than by team
    print("Randomizing starting positions...")
    env.unwrapped.civ_controller.ws_client.send_message("/set teamplacement DISABLED")
    time.sleep(0.5)

    # Toggle the connected player to be AI-controlled by Freeciv's built-in AI
    print(f"Toggling {fc_args['username']} to Freeciv AI control...")
    env.unwrapped.civ_controller.ws_client.send_message(f"/aitoggle {fc_args['username']}")
    time.sleep(1)

    # Aifill players are already AI-controlled by default (PLRF_AI flag set)
    # DO NOT toggle them - that would turn OFF their AI!
    print(f"All {num_ai_players - 1} aifill players are AI-controlled by default")

    done = False
    step = 0

    print(f"Game started - all {num_ai_players} players controlled by Freeciv AI")
    print(f"Running for up to {max_turns} turns (AI vs AI competitive game)")
    print()

    while not done:
        try:
            # NoOpAgent returns None, ending turn and letting Freeciv AI play
            action = agent.act(observations, info)
            observations, reward, terminated, truncated, info = env.step(action)

            turn = info.get('turn', 0)
            if turn > 0 and turn % 10 == 0:
                print(f"Turn {turn}/{max_turns}")

            step += 1
            # Environment will set terminated=True when max_turns is reached
            done = terminated or truncated

        except Exception as e:
            print(f"Error: {e}")
            raise e

    # Save and preserve the final game state for extracting complete production data
    # Autosave only happens at the beginning of turns, so we need to manually save at the end
    print("\nSaving final game state for complete data extraction...")
    env.unwrapped.civ_controller.save_game()
    env.unwrapped.civ_controller.delete_save = False  # Prevent deletion

    env.close()

    # Download and persist all savegames from Docker container
    print("\nDownloading savegames from Docker container...")
    recording_dir = f'logs/recordings/{username}'
    downloaded, skipped, failed = download_all_savegames_from_docker(username, recording_dir)
    print(f"Downloaded {downloaded} savegames (skipped {skipped} existing, {failed} failed)")

    print()
    print("="*60)
    print("DATA COLLECTION COMPLETE!")
    print("="*60)
    print()

    # Generate world reports automatically
    print("Generating world reports...")
    print()

    # Configuration for world report generation
    report_config = ReportConfig(
        # Input: where our game recording is stored
        recording_dir=f'logs/recordings/{username}/',

        # Output: where to save the report
        output_dir=f'reports/{username}/',

        # Generate report at the final turn
        report_turns=[max_turns],

        # Enable all implemented sections
        enabled_sections=['overview', 'historical_events', 'economics', 'demographics', 'technology'],

        # Output formats
        formats=['html'],

        # Visualization settings
        plot_style='seaborn',
        dpi=150
    )

    # Create generator and generate reports
    print("Initializing World Report Generator...")
    generator = ReportGenerator(report_config)

    print("Validating configuration...")
    if not generator.validate_config():
        print("Configuration validation failed!")
        return 1

    print("Configuration validated successfully!")
    print()
    print("Generating reports...")
    generator.generate_reports()

    print()
    print("="*60)
    print("WORLD REPORTS GENERATED!")
    print("="*60)
    print(f"\nReports saved to: {report_config.output_dir}")
    print("\nOpen the HTML files in your browser to view the reports.")

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run an all-AI Civilization game and generate world reports'
    )
    parser.add_argument(
        '--session_id',
        type=int,
        default=0,
        help='Session ID for isolating savegames (default: 0). Username will be myagent{session_id}'
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

    args = parser.parse_args()
    exit(main(
        session_id=args.session_id,
        max_turns=args.max_turns,
        num_ai_players=args.num_ai_players
    ))
