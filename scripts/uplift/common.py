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

def init_llm(model_id: str = "google/gemini-2.5-pro"):
    """Load .env + GCP secret keys and return a LiteLLMModel."""
    from dotenv import load_dotenv
    load_dotenv(CIVBENCH_ROOT / ".env")
    sys.path.insert(0, str(SRC))
    from civrealm.evaluation.models import load_api_keys_from_gcp, get_models
    load_api_keys_from_gcp()
    return get_models([model_id])[0]


def llm_ask(model, prompt: str, max_tokens: int = 4000, temperature: float = 0.7,
            retries: int = 4) -> str:
    """Call the model with retries. Gemini 2.5 spends tokens on hidden thinking,
    so keep max_tokens generous."""
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
    """Regex-parse a .sav.xz for the metrics we forecast/measure.

    Returns gold, government, score, alive, n_cities, n_units, known techs count,
    at_war (bool: player_id at war with anyone), turn.
    """
    c = _read_sav(path)
    out = {"turn": _int(re.search(r"\bturn=(\d+)", c))}

    pblock = _player_block(c, player_id)
    out["gold"] = _int(re.search(r"gold=(\d+)", pblock)) if pblock else None
    gm = re.search(r'government_name="([^"]*)"', pblock) if pblock else None
    out["government"] = gm.group(1) if gm else None
    out["score"] = _int(re.search(r'\bscore=(-?\d+)', pblock)) if pblock else None
    tech = re.search(r'research="inventions",?"?([01]+)"', pblock) if pblock else None
    out["n_techs"] = pblock and tech and tech.group(1).count("1") or 0

    # City / unit counts owned by player_id (savegame stores owner=N)
    out["n_cities"] = len(re.findall(rf'\n\[player{player_id}\.c\d+\]', c))
    out["n_units"] = len(re.findall(rf'\n\[player{player_id}\.u\d+\]', c))

    # Diplomacy: player at war. diplstate type 1 == WAR in freeciv (DS_WAR).
    out["at_war"] = _player_at_war(c, player_id)
    return out


def _player_block(content: str, pid: int) -> Optional[str]:
    """Return the text of [playerN] up to the next top-level [ header."""
    m = re.search(rf'\[player{pid}\]\n(.*?)(?=\n\[player{pid}\.|\n\[player{pid + 1}\]|\Z)',
                  content, flags=re.DOTALL)
    return m.group(0) if m else None


def _player_at_war(content: str, pid: int) -> Optional[bool]:
    """Detect DS_WAR in player pid's diplstates. Freeciv writes diplstates as a
    list; type 1 is war. We scan the player's dai/diplstate lines heuristically."""
    block = _player_block(content, pid)
    if block is None:
        return None
    # Common encodings: 'diplstate_type=...' vectors, or 'd_type="...,1,..."'
    war = re.search(r'diplstate.*?\b1\b', block, flags=re.DOTALL)
    return bool(war)


def _int(m):
    return int(m.group(1)) if m else None


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
