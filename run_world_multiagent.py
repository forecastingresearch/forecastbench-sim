#!/usr/bin/env python3
"""Multi-Agent Competitive Civilization Game with World Reports

This script runs a competitive game where multiple CivRealm agents play against
each other using Ray for parallel client connections. Each agent makes its own
autonomous decisions, creating a true competitive environment.

Setup:
- Configurable number of agents via AGENT_CONFIGS
- Each agent runs in its own Ray actor (separate process)
- All agents connect to the same multiplayer game server
- Agents make autonomous decisions using configured agent types
- Agent_0 serves as primary recorder for world report generation

Result: A competitive multi-agent game with full recording and world reports.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from civrealm.configs import fc_args
from civrealm.agents import ControllerAgent, RandomAgent, NoOpAgent
from civrealm.world_reports import ReportGenerator, ReportConfig
from civrealm.envs.freeciv_parallel_env import FreecivParallelEnv
from civrealm.freeciv.utils.port_utils import Ports
from civrealm.freeciv.utils.freeciv_logging import ray_logger_setup, fc_logger
import ray
import os

# Disable log deduplication of Ray to see messages from all actors
os.environ['RAY_DEDUP_LOGS'] = '0'

# =============================================================================
# CONFIGURATION - Modify this section to customize your experiment
# =============================================================================

# Agent configuration - easy to modify for different experiments
# Each entry defines: username, agent class, and random seed
# Note: Usernames follow pattern used in random_game_self_play.py (no underscores)
AGENT_CONFIGS = [
    {'username': 'agent', 'agent_type': ControllerAgent, 'seed': 12345},
    {'username': 'agent1', 'agent_type': ControllerAgent, 'seed': 67890},
    {'username': 'agent2', 'agent_type': ControllerAgent, 'seed': 11111},
    {'username': 'agent3', 'agent_type': ControllerAgent, 'seed': 22222},
    {'username': 'agent4', 'agent_type': ControllerAgent, 'seed': 33333},
]

# Game configuration
MAX_TURNS = 5  # Easy to adjust for different experiments

# Recording configuration
# First agent serves as primary recorder for world reports
PRIMARY_RECORDER = AGENT_CONFIGS[0]['username']

# =============================================================================
# MAIN IMPLEMENTATION
# =============================================================================

def main():
    """Run multi-agent competitive game with world report generation"""

    # Validate configuration
    num_agents = len(AGENT_CONFIGS)
    if num_agents < 2:
        raise ValueError("Need at least 2 agents for a competitive game")

    print("=" * 70)
    print("MULTI-AGENT COMPETITIVE CIVILIZATION GAME")
    print("=" * 70)
    print(f"\nNumber of agents: {num_agents}")
    print(f"Max turns: {MAX_TURNS}")
    print(f"Primary recorder: {PRIMARY_RECORDER}")
    print("\nAgent Configuration:")
    for i, config in enumerate(AGENT_CONFIGS):
        print(f"  {i}. {config['username']}: {config['agent_type'].__name__} (seed={config['seed']})")
    print()

    # Configure game parameters
    fc_args['max_turns'] = MAX_TURNS
    fc_args['minp'] = num_agents
    fc_args['self_play'] = True
    fc_args['multiplayer_game'] = True

    # Initialize Ray with logging
    ray.init(
        local_mode=False,
        runtime_env={"worker_process_setup_hook": ray_logger_setup},
    )
    logger = ray_logger_setup()

    # Get shared multiplayer game port
    print("Acquiring multiplayer game port...")
    client_port = Ports.get()
    print(f"Using port: {client_port}")
    print()

    # Create agents and environments
    print("Creating agents and environments...")
    agents = []
    envs = []

    for i, config in enumerate(AGENT_CONFIGS):
        # Create agent instance
        agent = config['agent_type']()
        agent.set_agent_seed(config['seed'])
        agents.append(agent)

        # Create parallel environment (Ray actor)
        env = FreecivParallelEnv.remote(
            'civrealm/FreecivBase-v0',
            username=config['username']
        )
        envs.append(env)

        print(f"  Created {config['username']}: {config['agent_type'].__name__}")

    print(f"\nAll {num_agents} agents created successfully!")
    print()

    # Initialize tracking structures
    observation_list = [None] * num_agents
    info_list = [None] * num_agents
    step_list = [0] * num_agents
    first_step = [True] * num_agents

    result_ids = []
    id_env_map = {}

    # Perform reset step for all environments
    print("Connecting agents to game server...")
    for i in range(num_agents):
        result_id = envs[i].reset.remote(
            client_port=client_port,
            minitask_pattern=None
        )
        result_ids.append(result_id)
        id_env_map[result_id] = i

    print("All agents connected! Game starting...")
    print()

    # Main game loop
    done = False
    turn_reported = -1

    while True:
        unready = result_ids.copy()
        unfinished = True

        while unfinished:
            # Wait for any agent to complete its action
            ready, unready = ray.wait(unready, num_returns=1)

            # Get the env id corresponding to the ready result
            env_id = id_env_map[ready[0]]
            del id_env_map[ready[0]]

            # Get username and result
            username = ray.get(envs[env_id].get_username.remote())
            result = ray.get(ready[0])

            # Debug: Check if result is None
            if result is None:
                print(f"ERROR: Result is None for {username} (env_id={env_id})")
                print(f"  ready[0] type: {type(ready[0])}")
                print(f"  first_step: {first_step[env_id]}")
                raise ValueError(f"Reset returned None for {username}")

            # Process result
            if first_step[env_id]:
                first_step[env_id] = False
                # Reset returns (observation, info)
                observation_list[env_id] = result[0]
                info_list[env_id] = result[1]
            else:
                # Step returns (observation, reward, terminated, truncated, info)
                observation_list[env_id] = result[0]
                reward = result[1]
                terminated = result[2]
                truncated = result[3]
                info_list[env_id] = result[4]

                current_turn = info_list[env_id].get('turn', 0)

                # Report progress every 10 turns
                if current_turn > 0 and current_turn != turn_reported and current_turn % 10 == 0:
                    print(f"Turn {current_turn}/{MAX_TURNS}")
                    turn_reported = current_turn

                step_list[env_id] += 1
                done = terminated or truncated

            # Agent decision loop - keep acting until agent ends turn
            while not done:
                try:
                    # Agent makes a decision
                    action = agents[env_id].act(
                        observation_list[env_id],
                        info_list[env_id]
                    )

                    # If action is None, agent ends turn
                    if action is None:
                        # End turn - this blocks until next turn starts
                        result_ids[env_id] = envs[env_id].step.remote(action)
                        id_env_map[result_ids[env_id]] = env_id
                        break
                    else:
                        # Execute normal action synchronously
                        result = ray.get(envs[env_id].step.remote(action))
                        observation_list[env_id] = result[0]
                        reward = result[1]
                        terminated = result[2]
                        truncated = result[3]
                        info_list[env_id] = result[4]

                        step_list[env_id] += 1
                        done = terminated or truncated

                except Exception as e:
                    fc_logger.error(f"Error for {username}: {repr(e)}")
                    raise e

            # If game is done, close this environment
            if done:
                ray.get(envs[env_id].close.remote())

            # Check if all agents have completed this round
            if not unready:
                unfinished = False

        # If game is done, break main loop
        if done:
            break

    print()
    print("=" * 70)
    print("GAME COMPLETE!")
    print("=" * 70)
    print()

    # Print final statistics for each agent
    for env_id in range(num_agents):
        username = AGENT_CONFIGS[env_id]['username']
        game_results = ray.get(envs[env_id].get_game_results.remote())
        print(f"{username}: {game_results}")

    print()
    print("=" * 70)
    print("GENERATING WORLD REPORTS")
    print("=" * 70)
    print()

    # Generate world reports from primary recorder
    print(f"Using recordings from: {PRIMARY_RECORDER}")
    print()

    # Check if recording directory exists
    import os
    recording_path = f'logs/recordings/{PRIMARY_RECORDER}/'
    if not os.path.exists(recording_path):
        print(f"WARNING: Recording directory not found: {recording_path}")
        print("Checking for alternative recording directories...")
        logs_path = 'logs/recordings/'
        if os.path.exists(logs_path):
            dirs = os.listdir(logs_path)
            print(f"Available recording directories: {dirs}")
            if dirs:
                # Use the first available directory
                alternative_recorder = dirs[0]
                recording_path = f'logs/recordings/{alternative_recorder}/'
                print(f"Using alternative recorder: {alternative_recorder}")
            else:
                print("ERROR: No recording directories found. Skipping world report generation.")
                return 0
        else:
            print("ERROR: logs/recordings/ directory does not exist. Skipping world report generation.")
            return 0

    # Configuration for world report generation
    report_config = ReportConfig(
        # Input: recorder's recordings (primary or alternative)
        recording_dir=recording_path,

        # Output: where to save the report
        output_dir='reports/latest_multiagent_game/',

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
    print("=" * 70)
    print("WORLD REPORTS GENERATED!")
    print("=" * 70)
    print(f"\nReports saved to: {report_config.output_dir}")
    print("\nOpen the HTML files in your browser to view the reports.")
    print("\nThe reports show data for ALL civilizations, not just the recorder.")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
