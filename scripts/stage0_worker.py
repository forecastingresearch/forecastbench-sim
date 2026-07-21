#!/usr/bin/env python3
"""Stage-0 world factory: for each seed, run the full per-world pipeline.

  1. run_world.py --seed N --max_turns <T>          -> logs/recordings/seedN/
  2. generate_data_batch.py --filter seedN          -> data/games_mc/seedN_data.json
  3. prepare_benchmark.py --game-id seedN           -> data/questions_mc/seedN/{questions.json, world_report/}
  4. mc_resolve.py (K reseeds + --baseline)         -> tmp/mc_resolve/seedN_h1.json

Every step is skipped if its output already exists (resumable). Steps run
sequentially per seed; run several workers on disjoint seed lists for
parallelism (each rollout/game occupies one freeciv-web port).

Usage:
  PYTHONPATH=src uv run python scripts/stage0_worker.py \
      --seeds 100 101 102 --mc-rollouts 20 --max-turns 62 \
      --log tmp/stage0/worker0.log
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def log(fh, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def run(cmd: list[str], fh, timeout: int | None = None) -> bool:
    log(fh, "RUN " + " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout)
    dt = time.time() - t0
    ok = proc.returncode == 0
    log(fh, f"  -> rc={proc.returncode} ({dt:.0f}s)")
    if not ok:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        log(fh, "  STDERR/OUT tail: " + tail.replace("\n", " | "))
    return ok


def world_done(seed: int) -> dict[str, bool]:
    gid = f"seed{seed}"
    rec = ROOT / "logs/recordings" / gid
    return {
        "recording": (rec / "savegames").exists() and any(
            f"T{60:03d}" in p.name or "T060" in p.name or "_T60" in p.name
            for p in (rec / "savegames").glob("*")),
        "data": (ROOT / f"data/games_mc/{gid}_data.json").exists(),
        "questions": (ROOT / f"data/questions_mc/{gid}/questions.json").exists(),
        "mc": _mc_valid(ROOT / f"tmp/mc_resolve/{gid}_h1.json"),
    }


def _mc_valid(path: Path) -> bool:
    """A killed mc_resolve leaves a partial json (baseline only, empty p_mc)."""
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError:
        path.unlink()
        return False
    if not d.get("p_mc"):
        path.unlink()
        return False
    return True


def process_seed(seed: int, mc_rollouts: int, max_turns: int, fh) -> bool:
    gid = f"seed{seed}"
    st = world_done(seed)
    log(fh, f"=== {gid} state={st}")

    py = [sys.executable]
    env_prefix = []  # PYTHONPATH set by caller

    if not st["recording"]:
        cmd = ["uv", "run", "python", "scripts/run_world.py",
               "--seed", str(seed), "--max_turns", str(max_turns), "--quiet"]
        # BeginTurnTimeoutException at reset is a transient contention failure
        if not run(cmd, fh, timeout=7200):
            log(fh, "  run_world failed; waiting 120s and retrying once")
            time.sleep(120)
            if not run(cmd, fh, timeout=7200):
                return False
    if not world_done(seed)["recording"]:
        log(fh, f"  !! no T60 savegame after run_world; aborting {gid}")
        return False

    if not st["data"]:
        if not run(["uv", "run", "python", "scripts/generate_data_batch.py",
                    "--input-dir", "logs/recordings",
                    "--output-dir", "data/games_mc",
                    "--filter", gid, "--workers", "1"], fh, timeout=1800):
            return False

    if not st["questions"]:
        if not run(["uv", "run", "python", "scripts/prepare_benchmark.py",
                    "--game-id", gid, "--snapshot-turn", "60",
                    "--games-dir", "data/games_mc",
                    "--output-dir", "data/questions_mc"], fh, timeout=1800):
            return False

    if not st["mc"]:
        seeds = [str(s) for s in range(1, mc_rollouts + 1)]
        if not run(["uv", "run", "python", "scripts/mc_resolve.py",
                    "--game-id", gid, "--base-seed", str(seed),
                    "--snapshot-turn", "60", "--resolution-turns", "90",
                    "--rng-seeds", *seeds, "--baseline",
                    "--questions", f"data/questions_mc/{gid}/questions.json",
                    "--output", f"tmp/mc_resolve/{gid}_h1.json"], fh,
                   timeout=3600 + 200 * mc_rollouts):
            return False

    # mc_resolve leaves ~73MB fork recording dirs per rollout; p_mc is already
    # extracted into the output json and forks are deterministically regenerable
    import shutil
    for d in (ROOT / "logs/recordings").glob(f"{gid}fork*"):
        shutil.rmtree(d, ignore_errors=True)

    # sanity: p_mc coverage
    mc = json.loads((ROOT / f"tmp/mc_resolve/{gid}_h1.json").read_text())
    n_q = len(mc.get("p_mc", {}))
    if n_q > 0:
        # per-turn states are ~150MB/world and fully superseded by the
        # extracted data json (savegames are kept for causal forks)
        for p in (ROOT / "logs/recordings" / gid).glob("turn_*_state.json"):
            p.unlink()
        for p in (ROOT / "logs/recordings" / gid).glob(
                "turn_*_available_action.json"):
            p.unlink()
    log(fh, f"=== {gid} DONE: {n_q} questions labeled, "
            f"n_rollouts={mc.get('n_rollouts')}")
    return n_q > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--mc-rollouts", type=int, default=20)
    ap.add_argument("--max-turns", type=int, default=62)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "tmp/mc_resolve").mkdir(parents=True, exist_ok=True)
    fh = open(args.log, "a")
    failures = []
    for seed in args.seeds:
        try:
            ok = process_seed(seed, args.mc_rollouts, args.max_turns, fh)
        except Exception as e:  # noqa: BLE001
            log(fh, f"  !! exception on seed{seed}: {e!r}")
            ok = False
        if not ok:
            failures.append(seed)
            log(fh, f"  seed{seed} FAILED, continuing with next seed")
    log(fh, f"WORKER DONE. failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
