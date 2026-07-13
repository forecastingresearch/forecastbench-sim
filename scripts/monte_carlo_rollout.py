#!/usr/bin/env python3
"""Monte Carlo rollouts from a single savegame.

Loads ONE savegame, reseeds Freeciv's RNG to N different values, runs each
forward K turns, and collects per-player outcomes. The point: replace
binary "did event E happen?" resolutions with continuous frequencies
across rollouts, which gives a real-valued resolution signal usable
for RLVR-style training.

Two methods are supported (and intended to match):

  --method save_edit  (Method B, civbench-only)
      Pre-load: edit the savegame's [random] block to the state Freeciv
      would have after fc_srand(rng_seed). Pure Python, no Freeciv patch.

  --method server_reseed  (Method A, requires Freeciv patch)
      Use the vanilla savegame. Before each load, /set gameseed N and
      rely on a patched sg_load_random that re-applies fc_srand after
      restoring the table. Requires applying patches/freeciv_reseed.patch
      to the freeciv-web docker server.

Usage:
    uv run python scripts/monte_carlo_rollout.py \\
        --base-seed 122 --checkpoint-turn 100 --run-turns 5 \\
        --rng-seeds 1 2 3 4 5 --method save_edit \\
        --output tmp/mc/run.json
"""
from __future__ import annotations

import sys
import json
import argparse
import subprocess
import time
from pathlib import Path

# Parse our args first, before importing civrealm (it has its own argparse).
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--base-seed", type=int, required=True)
_parser.add_argument("--checkpoint-turn", type=int, required=True)
_parser.add_argument("--run-turns", type=int, required=True)
_parser.add_argument(
    "--rng-seeds", type=int, nargs="+", required=True,
    help="One Freeciv RNG seed per rollout."
)
_parser.add_argument(
    "--method", choices=["save_edit", "server_reseed"], default="save_edit",
)
_parser.add_argument("--recording-dir", type=str, default=None)
_parser.add_argument("--output", type=str, required=True)
_parser.add_argument("--tag", type=str, default="mc",
                     help="Used in fork usernames to keep runs isolated.")
_pre_args, _ = _parser.parse_known_args()


CIVBENCH_ROOT = Path(__file__).parent.parent


def run_one_rollout(
    recording_dir: str,
    base_seed: int,
    checkpoint_turn: int,
    end_turn: int,
    rng_seed: int,
    method: str,
    fork_name: str,
) -> dict:
    """Run one rollout in a subprocess, return its result dict.

    Subprocess isolation is required: civrealm has global singleton state
    (Ports, fc_args) that breaks the second run-in-process.
    """
    if method == "save_edit":
        modifications = [{"type": "rng_seed", "value": rng_seed}]
        pre_load_hook = ""
    elif method == "server_reseed":
        # No save mutation. Instead, write /tmp/freeciv_reseed inside the
        # docker container; the patched sg_load_random reads + consumes
        # this file when the save is loaded (see patches/freeciv_reseed.patch).
        modifications = []
        pre_load_hook = (
            '\nimport subprocess\n'
            f'subprocess.run(["docker", "exec", "freeciv-web", "bash", "-c",'
            f' "echo {rng_seed} > /tmp/freeciv_reseed"], check=True)\n'
        )
    else:
        raise ValueError(method)

    script = f'''
import sys, json, traceback
from pathlib import Path
sys.path.insert(0, str(Path(".") / "src"))
{pre_load_hook}
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
        "error": result.error,
    }}
except Exception as e:
    output = {{"success": False, "final_turn": 0, "player_states": {{}}, "error": traceback.format_exc()}}
print("RESULT_JSON:" + json.dumps(output))
'''
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(CIVBENCH_ROOT),
    )
    for line in proc.stdout.split("\n"):
        if line.startswith("RESULT_JSON:"):
            data = json.loads(line[len("RESULT_JSON:"):])
            data["rng_seed"] = rng_seed
            data["fork_name"] = fork_name
            return data
    return {
        "rng_seed": rng_seed,
        "fork_name": fork_name,
        "success": False,
        "final_turn": 0,
        "player_states": {},
        "error": (
            f"Subprocess produced no RESULT_JSON. "
            f"stderr tail: {proc.stderr[-1000:]!r}"
        ),
    }


def main() -> int:
    args = _pre_args
    recording_dir = args.recording_dir or f"logs/recordings/s{args.base_seed}"
    if not (CIVBENCH_ROOT / recording_dir).exists():
        print(f"Recording not found: {recording_dir}", file=sys.stderr)
        return 1

    end_turn = args.checkpoint_turn + args.run_turns
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"== Monte Carlo rollout ==")
    print(f"  recording        : {recording_dir}")
    print(f"  checkpoint turn  : {args.checkpoint_turn}")
    print(f"  end turn         : {end_turn}")
    print(f"  N rollouts       : {len(args.rng_seeds)}")
    print(f"  method           : {args.method}")
    print(f"  output           : {out_path}")
    print()

    results = []
    t_start = time.time()
    for i, rng_seed in enumerate(args.rng_seeds):
        # Fork name must be filesystem-safe and underscore-free (Freeciv
        # splits filenames on '_' to extract the host_name on load).
        fork_name = f"{args.tag}{args.method[:3]}s{rng_seed}"
        t0 = time.time()
        print(f"[{i + 1}/{len(args.rng_seeds)}] seed={rng_seed} name={fork_name}")
        r = run_one_rollout(
            recording_dir=recording_dir,
            base_seed=args.base_seed,
            checkpoint_turn=args.checkpoint_turn,
            end_turn=end_turn,
            rng_seed=rng_seed,
            method=args.method,
            fork_name=fork_name,
        )
        dt = time.time() - t0
        ok = "OK" if r["success"] else "FAIL"
        scores = {pid: s.get("score") for pid, s in r.get("player_states", {}).items()}
        print(f"     -> {ok} ({dt:.1f}s) final_turn={r['final_turn']} scores={scores}")
        if not r["success"]:
            print(f"     ! error: {(r.get('error') or '')[:300]}")
        results.append(r)

        # Periodically flush partial results so a crash mid-run isn't fatal.
        out_path.write_text(json.dumps({
            "config": vars(args),
            "results": results,
            "elapsed_sec": time.time() - t_start,
            "in_progress": (i + 1) < len(args.rng_seeds),
        }, indent=2))

    print()
    print(f"Done in {time.time() - t_start:.1f}s. Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
