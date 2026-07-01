"""Smoke test: does the Gemini-ported BaseLang agent actually play FreeCiv?
Runs a few turns on a fresh game and logs actions + gold. Not the real A/B."""
import sys, time, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "llm_baseline"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import gymnasium
import civrealm  # noqa: F401  (registers envs)
from civrealm.configs import fc_args
from civrealm.evaluation.models import load_api_keys_from_gcp
from agents import BaseLangAgent

load_api_keys_from_gcp()
fc_args["max_turns"] = int(os.environ.get("SMOKE_MAX_TURNS", "3"))
fc_args["aifill"] = 5

print("creating env FreecivLLM-v0 ...", flush=True)
env = gymnasium.make("civrealm/FreecivLLM-v0")
agent = BaseLangAgent()
obs, info = env.reset()
print(f"reset done. turn={info.get('turn')}", flush=True)

def gold():
    try:
        pc = env.unwrapped.civ_controller.player_ctrl
        return pc.my_player.get("gold")
    except Exception as e:
        return f"err:{e}"

done = False
step = 0
t0 = time.time()
n_actions = 0
while not done and step < 400 and (time.time() - t0) < 600:
    action = agent.act(obs, info)
    obs, reward, terminated, truncated, info = env.step(action)
    if action is not None:
        n_actions += 1
    step += 1
    if step % 10 == 0:
        print(f"step={step} turn={info.get('turn')} gold={gold()} last_action={action}", flush=True)
    done = terminated or truncated

print(f"DONE steps={step} turns={info.get('turn')} actions_taken={n_actions} "
      f"final_gold={gold()} elapsed={time.time()-t0:.0f}s", flush=True)
env.close()
