"""Track A: run ONE free-play game with the Gemini BaseLang agent, controlling
player 0, with the explicit objective of maximizing gold by turn H. Two arms:

  control  : objective only
  forecast : objective + a game-true p_mc forecast table (re-shown every turn,
             prepended to every actor prompt)
  scrambled: objective + the same probabilities shuffled across events

Reads final gold live from the controller. One game per process (civrealm keeps
global singleton state). Meant to be launched in parallel by agentic_ab.py.

Usage:
  PYTHONPATH=src:llm_baseline uv run python scripts/uplift/agentic_run.py \
    --seed 100 --arm forecast --horizon 28 --ai 5 \
    --forecasts data/uplift/forecasts_agentic.json --output tmp/uplift/ag_100_forecast.json
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "llm_baseline"))
sys.path.insert(0, str(ROOT / "scripts" / "uplift"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from dv_decide import EVENT_LABELS, forecast_table  # noqa: E402


def make_mission(objective: str, forecast_block: str):
    import gymnasium  # noqa: F401
    import civrealm  # noqa: F401
    from agents import BaseLangAgent

    class MissionAgent(BaseLangAgent):
        def get_obs_input_prompt(self, ctrl_type, actor_name, actor_dict, available_actions):
            base = super().get_obs_input_prompt(ctrl_type, actor_name, actor_dict, available_actions)
            preamble = f"MISSION: {objective}\n"
            if forecast_block:
                preamble += forecast_block + "\n"
            return preamble + "\n" + base
    return MissionAgent()


def build_forecast_block(fc_path, seed, arm, rng):
    if arm == "control" or not fc_path:
        return ""
    bank = json.loads(Path(fc_path).read_text())
    sc = next((s for s in bank["scenarios"] if s["seed"] == seed), None)
    if not sc:
        return ""
    ev_keys = [e for e in EVENT_LABELS
               if e in sc["p_mc"][f"h{sc['fc_horizons'][-1]}"]]
    if arm == "scrambled":
        shuffled = ev_keys[:]
        rng.shuffle(shuffled)
        order = list(zip(ev_keys, shuffled))
    else:
        order = [(e, e) for e in ev_keys]
    table = forecast_table(sc["p_mc"], sc["fc_horizons"], order)
    return ("Monte-Carlo forecasts (game-true probabilities from simulating this "
            "position forward many times):\n" + table)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", choices=["control", "forecast", "scrambled"], required=True)
    ap.add_argument("--horizon", type=int, default=28, help="max turns to play")
    ap.add_argument("--ai", type=int, default=5)
    ap.add_argument("--forecasts", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--max-seconds", type=int, default=5400)
    args = ap.parse_args()

    if args.model:
        os.environ["UPLIFT_MODEL"] = args.model

    import gymnasium
    import civrealm  # noqa: F401
    from civrealm.configs import fc_args
    from civrealm.evaluation.models import load_api_keys_from_gcp
    load_api_keys_from_gcp()

    # Deterministic fresh game for this seed (same world as the base game used to
    # compute forecasts).
    fc_args["username"] = f"ag{args.seed}{args.arm[:3]}"
    fc_args["max_turns"] = args.horizon
    fc_args["aifill"] = args.ai
    fc_args["debug.record_action_and_observation"] = False
    fc_args["debug.randomly_generate_seeds"] = False
    fc_args["debug.mapseed"] = args.seed
    fc_args["debug.gameseed"] = args.seed
    random.seed(args.seed)

    objective = (f"You are player 0 in FreeCiv. Your SOLE objective is to maximize "
                 f"your treasury (GOLD) by turn {args.horizon}. Prioritize actions "
                 f"that grow and protect your economy; nothing else matters.")
    fc_block = build_forecast_block(args.forecasts, args.seed, args.arm,
                                    random.Random((args.seed, args.arm)))

    env = gymnasium.make("civrealm/FreecivLLM-v0")
    agent = make_mission(objective, fc_block)
    obs, info = env.reset()

    def gold():
        try:
            return env.unwrapped.civ_controller.player_ctrl.my_player.get("gold")
        except Exception:  # noqa: BLE001
            return None

    gold_trace = {}
    done, step, t0 = False, 0, time.time()
    while not done and step < args.max_steps and (time.time() - t0) < args.max_seconds:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        t = info.get("turn")
        if t is not None:
            gold_trace[t] = gold()
        step += 1
        done = terminated or truncated

    final_gold = gold()
    result = {
        "seed": args.seed, "arm": args.arm, "horizon": args.horizon,
        "final_gold": final_gold, "final_turn": info.get("turn"),
        "steps": step, "elapsed_sec": round(time.time() - t0, 1),
        "gold_trace": gold_trace,
        "had_forecast": bool(fc_block),
    }
    try:
        env.close()
    except Exception:  # noqa: BLE001
        pass
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print("RESULT:" + json.dumps({k: result[k] for k in
          ("seed", "arm", "final_gold", "final_turn", "steps", "elapsed_sec")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
