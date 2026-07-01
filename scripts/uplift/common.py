"""Shared infrastructure for the forecast-uplift study.

Two tracks use this:
  - Track B (decision-value): fork a checkpoint with a chosen policy, roll forward
    under the FreeCiv AI, read the resulting gold. Also compute game-true p_mc
    forecasts by rolling the baseline many times and resolving events.
  - Track A (agentic play): reuses the LLM helper + gold/savegame parsing.

Rollouts run in a subprocess (civrealm keeps global singleton state that breaks a
second in-process game), mirroring scripts/monte_carlo_rollout.py.
"""
from __future__ import annotations

import glob
import json
import lzma
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

CIVBENCH_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = CIVBENCH_ROOT / "src"

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.environ.get("UPLIFT_MODEL", "google/gemini-2.5-flash")


def init_llm(model_id: str = DEFAULT_MODEL):
    """Load .env + GCP secret keys and return a LiteLLMModel."""
    from dotenv import load_dotenv
    load_dotenv(CIVBENCH_ROOT / ".env")
    sys.path.insert(0, str(SRC))
    from civrealm.evaluation.models import load_api_keys_from_gcp, get_models
    load_api_keys_from_gcp()
    return get_models([model_id])[0]


def llm_ask(model, prompt: str, max_tokens=None, temperature: float = 0.7,
            retries: int = 5) -> str:
    """Call the model with retries. max_tokens=None => no output cap (we never
    truncate model outputs; Gemini 2.5 also spends tokens on hidden thinking)."""
    last = None
    for i in range(retries):
        try:
            out = model.get_response(prompt, temperature=temperature, max_tokens=max_tokens)
            if out and out.strip():
                return out
            last = "empty response"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"LLM failed after {retries} tries: {last}")


# ---------------------------------------------------------------------------
# Rollout with arbitrary policy modifications
# ---------------------------------------------------------------------------

_ROLLOUT_TEMPLATE = '''
import sys, json, traceback
from pathlib import Path
sys.path.insert(0, str(Path({root!r}) / "src"))
from civrealm.forking import ForkManager
try:
    manager = ForkManager({recording_dir!r}, {base_seed})
    fork = manager.create_fork(
        checkpoint_turn={checkpoint_turn},
        modifications={modifications!r},
        fork_name={fork_name!r},
    )
    result = manager.run_fork(fork, {end_turn})
    output = {{
        "success": result.success,
        "final_turn": result.final_turn,
        "player_states": result.player_states,
        "output_dir": fork.output_dir,
        "error": result.error,
    }}
except Exception:
    output = {{"success": False, "final_turn": 0, "player_states": {{}},
              "output_dir": None, "error": traceback.format_exc()}}
print("RESULT_JSON:" + json.dumps(output))
'''


def rollout(recording_dir: str, base_seed: int, checkpoint_turn: int, end_turn: int,
            rng_seed: int, fork_name: str, extra_mods: Optional[list] = None,
            player_id: int = 0, timeout: int = 900) -> dict:
    """Fork `recording_dir` at `checkpoint_turn`, apply rng reseed + extra_mods,
    run to `end_turn` under the AI, return result incl. path to final savegame.

    extra_mods e.g. [{"type": "government", "player_id": 0, "value": "Republic"}].
    """
    mods = [{"type": "rng_seed", "value": rng_seed}]
    if extra_mods:
        mods = list(extra_mods) + mods  # apply policy first, reseed last
    script = _ROLLOUT_TEMPLATE.format(
        root=str(CIVBENCH_ROOT), recording_dir=recording_dir, base_seed=base_seed,
        checkpoint_turn=checkpoint_turn, end_turn=end_turn,
        modifications=mods, fork_name=fork_name,
    )
    try:
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, cwd=str(CIVBENCH_ROOT), timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"rng_seed": rng_seed, "fork_name": fork_name, "success": False,
                "final_turn": 0, "player_states": {}, "final_savegame": None,
                "error": f"timeout after {timeout}s"}
    data = None
    for line in proc.stdout.split("\n"):
        if line.startswith("RESULT_JSON:"):
            data = json.loads(line[len("RESULT_JSON:"):])
            break
    if data is None:
        return {"rng_seed": rng_seed, "fork_name": fork_name, "success": False,
                "final_turn": 0, "player_states": {}, "final_savegame": None,
                "error": f"no RESULT_JSON. stderr tail: {proc.stderr[-800:]!r}"}
    data["rng_seed"] = rng_seed
    data["fork_name"] = fork_name
    data["final_savegame"] = None
    if data.get("output_dir"):
        data["final_savegame"] = _find_final_savegame(data["output_dir"], data.get("final_turn", end_turn))
    return data


def _find_final_savegame(output_dir: str, target_turn: int) -> Optional[str]:
    """Pick the downloaded savegame with the highest turn <= target_turn."""
    cands = glob.glob(str(Path(output_dir) / "*.sav.xz"))
    best, best_turn = None, -1
    for c in cands:
        m = re.search(r"_T(\d+)_", Path(c).name)
        if not m:
            continue
        t = int(m.group(1))
        if t <= target_turn and t > best_turn:
            best, best_turn = c, t
    return best


# ---------------------------------------------------------------------------
# Savegame parsing (rich end-state for events + outcome)
# ---------------------------------------------------------------------------

def _read_sav(path: str) -> str:
    with lzma.open(path, "rt") as f:
        return f.read()


def parse_savegame(path: str, player_id: int = 0) -> dict:
    """Regex-parse a .sav.xz (freeciv savegame3 format) for the quantities we
    forecast/measure: gold, government, n_cities, n_units, at_war, turn.

    Cities/units are stored inside the [playerN] section as ncities=/nunits=.
    Diplomacy is a diplstate table whose first column ("current") is "War" when
    at war. score is read from run_fork's player_states, not here.
    """
    c = _read_sav(path)
    out = {"turn": _int(re.search(r"\bturn=(\d+)", c))}

    pblock = _player_block(c, player_id)
    if pblock:
        out["gold"] = _int(re.search(r"\ngold=(\d+)", pblock))
        gm = re.search(r'government_name="([^"]*)"', pblock)
        out["government"] = gm.group(1) if gm else None
        out["n_cities"] = _int(re.search(r"\nncities=(\d+)", pblock))
        out["n_units"] = _int(re.search(r"\nnunits=(\d+)", pblock))
        out["at_war"] = _player_at_war(pblock)
    else:
        out.update(gold=None, government=None, n_cities=None, n_units=None, at_war=None)
    return out


def _player_block(content: str, pid: int) -> Optional[str]:
    """Return the full [playerN] section (incl. its city/unit tables) up to the
    next [player{N+1}] header (or EOF)."""
    start = content.find(f"[player{pid}]")
    if start == -1:
        return None
    end = content.find(f"[player{pid + 1}]", start)
    return content[start:] if end == -1 else content[start:end]


def _player_at_war(pblock: str) -> Optional[bool]:
    """A player is at war if any diplstate row's first ("current") column is War."""
    dip = re.search(r"diplstate=\{.*?\n\}", pblock, flags=re.DOTALL)
    if not dip:
        return None
    # Each data row looks like:  "War","Cease-fire",31,0,...  -> current == War
    return bool(re.search(r'^"War",', dip.group(0), flags=re.MULTILINE))


def _int(m):
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# State-JSON parsing (richer + structured; used for event resolution)
# ---------------------------------------------------------------------------

DS_WAR = 1  # freeciv diplomatic_state code for war


def find_state_json(output_dir: str, turn: int) -> Optional[str]:
    """Path to the recorded state JSON at the given turn (or the closest earlier
    turn) inside a fork's output dir."""
    best, best_turn = None, -1
    for c in glob.glob(str(Path(output_dir) / "turn_*_state.json")):
        m = re.search(r"turn_0*(\d+)_", Path(c).name)
        if not m:
            continue
        t = int(m.group(1))
        if t <= turn and t > best_turn:
            best, best_turn = c, t
    return best


def parse_state_json(path: str, player_id: int = 0) -> dict:
    """Extract the quantities we forecast/measure from a recorded state JSON.

    Returns: turn, gold, score, government_name, is_alive, n_cities, n_units,
    at_war (player_id at war with anyone), rival_max_score, top_score (bool),
    scores (all players)."""
    s = json.load(open(path))
    pid = str(player_id)
    players = s.get("player", {})
    p = players.get(pid, {})
    scores = {k: v.get("score") for k, v in players.items()
              if isinstance(v, dict) and v.get("score") is not None}
    rival = [v for k, v in scores.items() if k != pid]
    my_score = scores.get(pid)
    n_cities = sum(1 for c in s.get("city", {}).values()
                   if isinstance(c, dict) and str(c.get("owner")) == pid)
    n_units = sum(1 for u in s.get("unit", {}).values()
                  if isinstance(u, dict) and str(u.get("owner")) == pid)
    dipl = s.get("dipl", {})
    at_war = any(isinstance(v, dict) and v.get("diplomatic_state") == DS_WAR
                 and k != pid for k, v in dipl.items())
    return {
        "turn": s.get("game", {}).get("turn") or _turn_from_name(path),
        "gold": p.get("gold"),
        "score": my_score,
        "government_name": p.get("government_name"),
        "is_alive": p.get("is_alive"),
        "n_cities": n_cities,
        "n_units": n_units,
        "at_war": at_war,
        "rival_max_score": max(rival) if rival else None,
        "top_score": (my_score is not None and rival != [] and my_score >= max(rival)),
        "scores": scores,
    }


def _turn_from_name(path: str) -> Optional[int]:
    m = re.search(r"turn_0*(\d+)_", Path(path).name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Parallel rollout execution
# ---------------------------------------------------------------------------

def run_rollouts_parallel(specs: list, max_workers: int = 6) -> list:
    """Run many rollouts concurrently. Each spec is a kwargs dict for rollout().
    civrealm coordinates freeciv ports across processes via a filelock, so
    independent subprocesses safely claim distinct ports (6001-6009).
    Returns results in the same order as specs."""
    from concurrent.futures import ThreadPoolExecutor
    results = [None] * len(specs)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(rollout, **spec): i for i, spec in enumerate(specs)}
        for fut in futs:
            pass
        for fut, i in futs.items():
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[i] = {"success": False, "error": f"executor: {e!r}",
                              "player_states": {}, "output_dir": None}
    return results


def final_gold(res: dict, player_id: int = 0) -> Optional[float]:
    """Gold from player_states (preferred) else from the final savegame."""
    ps = res.get("player_states") or {}
    for key in (player_id, str(player_id)):
        if key in ps and ps[key].get("gold") is not None:
            return float(ps[key]["gold"])
    if res.get("final_savegame"):
        try:
            return float(parse_savegame(res["final_savegame"], player_id).get("gold"))
        except Exception:  # noqa: BLE001
            return None
    return None
