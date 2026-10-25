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
from pathlib import Path

# Add src to path for world report imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from civrealm.configs import fc_args
from civrealm.agents import NoOpAgent, ObserverAgent
from civrealm.world_reports import ReportGenerator, ReportConfig
import gymnasium
import time
import threading

# Configuration
# this username is for authenticating to the game server
# we pull the civilization name for world reports
fc_args['username'] = 'myagent2'
fc_args['debug.record_action_and_observation'] = True

# Game Configuration
MAX_TURNS = 50  # Full game with savegame-based complete data extraction
fc_args['max_turns'] = MAX_TURNS

# Don't wait for observer - let observer join dynamically
# Observer will connect after game starts
fc_args['wait_for_observer'] = False

# AI Configuration
# NOTE: aifill sets TOTAL number of players in game (not additional AI opponents)
# So aifill=5 creates 4 AI players + 1 connected player = 5 total
NUM_AI_PLAYERS = 5  # TOTAL players in game
AI_DIFFICULTY = 'hard'  # Options: 'handicapped', 'novice', 'easy', 'normal', 'hard', 'cheating', 'experimental'
fc_args['aifill'] = NUM_AI_PLAYERS


def run_observer(client_port):
    """Run observer client to record complete game state

    Observer sees complete state (no fog of war) for all players,
    enabling accurate world report generation.

    Args:
        client_port: Port to connect to (same game as player)
    """
    print(f"\n[Observer] Starting observer connection...")

    # Temporarily modify fc_args for observer
    # Save original values
    original_username = fc_args['username']
    original_self_play = fc_args['self_play']

    # Set observer-specific values
    fc_args['username'] = OBSERVER_USERNAME
    fc_args['self_play'] = True  # Join the same game as player

    # Create observer environment
    observer_env = gymnasium.make('civrealm/FreecivBase-v0')
    observer_agent = ObserverAgent()

    # Restore original fc_args
    fc_args['username'] = original_username
    fc_args['self_play'] = original_self_play

    try:
        # Connect to the same game server by passing client_port to reset
        observations, info = observer_env.reset(client_port=client_port)
        print(f"[Observer] Connected successfully!")

        # Send /observe command to enter observer mode (see complete game state)
        print(f"[Observer] Entering observer mode with /observe command...")
        observer_env.unwrapped.civ_controller.ws_client.send_message("/observe")
        time.sleep(1)
        print(f"[Observer] Observer mode active! Recording to: logs/recordings/{OBSERVER_USERNAME}/")

        # Observer loop - just observe, don't take actions
        done = False
        while not done:
            action = observer_agent.act(observations, info)
            observations, reward, terminated, truncated, info = observer_env.step(action)

            turn = info.get('turn', 0)
            if turn > 0 and turn % 50 == 0:
                print(f"[Observer] Recording turn {turn}")

            done = terminated or truncated

        observer_env.close()
        print("[Observer] Recording complete")

    except Exception as e:
        print(f"[Observer] Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("Starting all-AI game collection...")
    print(f"Username: {fc_args['username']} (will be toggled to AI control)")
    print(f"AI Players: {NUM_AI_PLAYERS} total (all Freeciv AI at {AI_DIFFICULTY} difficulty)")
    print(f"Setup: {NUM_AI_PLAYERS - 1} via aifill + 1 connected player toggled to AI")
    print(f"Max turns: {MAX_TURNS}")
    print(f"Recording to: logs/recordings/{fc_args['username']}/")


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
    print(f"All {NUM_AI_PLAYERS - 1} aifill players are AI-controlled by default")

    # Start observer in separate thread if enabled (AFTER game setup)
    observer_thread = None


    done = False
    step = 0

    print(f"Game started - all {NUM_AI_PLAYERS} players controlled by Freeciv AI")
    print(f"Running for up to {MAX_TURNS} turns (AI vs AI competitive game)")
    print()

    while not done:
        try:
            # NoOpAgent returns None, ending turn and letting Freeciv AI play
            action = agent.act(observations, info)
            observations, reward, terminated, truncated, info = env.step(action)

            turn = info.get('turn', 0)
            if turn > 0 and turn % 10 == 0:
                print(f"Turn {turn}/{MAX_TURNS}")

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

    print()
    print("="*60)
    print("DATA COLLECTION COMPLETE!")
    print("="*60)
    print()

    # Generate world reports automatically
    print("Generating world reports...")
    print()

    # Configuration for world report generation
    # Use observer recordings if available (complete state without fog of war)
    # Otherwise fall back to player recordings (limited by fog of war)
    recording_source = fc_args["username"]

    report_config = ReportConfig(
        # Input: where our game recording is stored
        recording_dir=f'logs/recordings/{recording_source}/',

        # Output: where to save the report
        output_dir='reports/latest_game/',

        # Generate report at the final turn
        report_turns=[MAX_TURNS],

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
    main()
