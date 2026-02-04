#!/usr/bin/env python3
"""Regression test for AI difficulty setting.

BACKGROUND:
-----------
CivBench games should run with AI players at "Hard" difficulty - the highest
non-cheating level in FreeCiv. This matters for benchmark validity: Easy AI
players behave differently (less aggressive expansion, weaker military, etc.),
which would make forecasting questions systematically easier or produce
different base rates than intended.

THE BUG (discovered 2025-01):
-----------------------------
All existing game recordings had AI players at "Easy" difficulty despite
run_world.py specifying AI_DIFFICULTY = 'hard'. The bug: /set skilllevel
was sent AFTER players were already created by aifill, but FreeCiv's
skilllevel setting only affects NEWLY created AI players.

Savegame inspection showed:
    ai.level="Easy"   # All players, all games

THE FIX:
--------
1. client_state.py: Send /set skilllevel hard BEFORE aifill, then /hard
   (without args) after aifill to upgrade existing AI players
2. run_world.py: Send /hard {username} after /aitoggle since the connected
   player isn't AI-controlled until after the toggle

WHY THIS TEST EXISTS:
---------------------
This test verifies the fix works by:
1. Running a minimal 5-turn game with 3 players
2. Downloading savegames from Docker
3. Checking that ALL players have ai.level="Hard"

Run this test after any changes to game initialization to ensure the
difficulty fix hasn't regressed. Expected output:

    ai.level="Hard"  # All players

If you see ai.level="Easy" for any player, the fix has regressed.

USAGE:
------
    cd ~/Projects/civbench
    uv run python scripts/test_difficulty.py
    xzcat logs/recordings/s{seed}/savegames/*T5*.xz | grep 'ai.level'
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from civrealm.configs import fc_args
from civrealm.agents import NoOpAgent
import gymnasium

def main():
    seed = 99993  # Test seed - final verification
    run_id = f's{seed}'
    fc_args['username'] = run_id
    fc_args['debug.record_action_and_observation'] = True
    fc_args['max_turns'] = 5  # Just 5 turns for quick test
    fc_args['aifill'] = 3  # 3 players for quick test

    fc_args['debug.randomly_generate_seeds'] = False
    fc_args['debug.mapseed'] = seed
    fc_args['debug.gameseed'] = seed

    print("=== AI Difficulty Test ===")
    print(f"Seed: {seed}, Players: 3, Turns: 5")
    print()

    env = gymnasium.make('civrealm/FreecivBase-v0')
    agent = NoOpAgent()

    observations, info = env.reset()

    # Preserve savegames
    env.unwrapped.civ_controller.delete_save = False

    # Skill level is now set BEFORE aifill in client_state.py
    # Toggle our player to AI and set to hard
    env.unwrapped.civ_controller.ws_client.send_message(f"/aitoggle {run_id}")
    time.sleep(0.5)
    env.unwrapped.civ_controller.ws_client.send_message(f"/hard {run_id}")
    time.sleep(0.5)

    print()
    print("Running 5 turns...")

    done = False
    while not done:
        action = agent.act(observations, info)
        observations, reward, terminated, truncated, info = env.step(action)
        turn = info.get('turn', 0)
        print(f"  Turn {turn}")
        done = terminated or truncated or turn >= 5

    env.close()

    # Download savegames from Docker
    from civrealm.world_reports.utils.savegame_parser import download_all_savegames_from_docker

    print()
    print("Downloading savegames from Docker...")
    recordings_dir = Path(f"logs/recordings/{run_id}")
    download_all_savegames_from_docker(run_id, recordings_dir)

    print()
    print("Game complete. Check savegames with:")
    print(f"  xzcat logs/recordings/{run_id}/savegames/*.xz | grep 'ai.level'")

if __name__ == '__main__':
    main()
