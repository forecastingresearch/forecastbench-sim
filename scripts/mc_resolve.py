#!/usr/bin/env python3
"""Monte Carlo *resolution* of real civbench questions.

For one game (e.g. seed0), starts at the world-report turn (snapshot_turn),
runs N rollouts to the resolution turn(s) with different Freeciv RNG seeds,
serializes each rollout, and resolves the game's existing questions against
each rollout. Averaging the boolean answers across rollouts yields a
real-valued resolution p_mc in [0, 1] instead of a hard 0/1.

This is the "real answer" within the simulated world: the frequency the
event occurs across many futures branched from the same observed state.

Prereq: a start savegame at snapshot_turn must exist in
logs/recordings/<game_id>/savegames/. Regenerate deterministically with:
    PYTHONPATH=src uv run python scripts/run_world.py --seed N --max_turns 62

Usage:
    PYTHONPATH=src uv run python scripts/mc_resolve.py \\
        --game-id seed0 --base-seed 0 \\
        --snapshot-turn 60 --resolution-turns 90 \\
        --rng-seeds 1 2 3 4 5 6 7 8 \\
        --questions data/questions/seed0/questions.json \\
        --output tmp/mc_resolve/seed0_h1.json
"""
from __future__ import annotations

import sys
import json
import argparse
import subprocess
import time
from pathlib import Path

sys.path.insert(0, "src")

CIVBENCH_ROOT = Path(__file__).parent.parent


def _pre_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--game-id", required=True)
    p.add_argument("--base-seed", type=int, required=True)
    p.add_argument("--snapshot-turn", type=int, required=True)
    p.add_argument("--resolution-turns", type=int, nargs="+", required=True)
    p.add_argument("--rng-seeds", type=int, nargs="+", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--baseline", action="store_true",
                   help="Also run one rollout with the savegame's original RNG (no reseed).")
    a, _ = p.parse_known_args()
    return a


def run_rollout_subprocess(game_id: str, base_seed: int, snapshot_turn: int,
                           end_turn: int, rng_seed: int | None, fork_name: str) -> dict:
    """Run one fork to end_turn. rng_seed=None -> baseline (no reseed).
    Returns {success, output_dir, error}."""
    recording_dir = f"logs/recordings/{game_id}"
    if rng_seed is None:
        modifications = []
    else:
        modifications = [{"type": "rng_seed", "value": rng_seed}]

    script = f'''
import sys, json, traceback
from pathlib import Path
sys.path.insert(0, str(Path(".") / "src"))
from civrealm.forking import ForkManager
try:
    mgr = ForkManager({recording_dir!r}, {base_seed})
    fork = mgr.create_fork(checkpoint_turn={snapshot_turn},
                           modifications={modifications!r},
                           fork_name={fork_name!r})
    res = mgr.run_fork(fork, {end_turn})
    out = {{"success": res.success, "output_dir": fork.output_dir,
            "final_turn": res.final_turn, "error": res.error}}
except Exception:
    out = {{"success": False, "output_dir": None, "final_turn": 0,
            "error": traceback.format_exc()}}
print("RESULT_JSON:" + json.dumps(out))
'''
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, cwd=str(CIVBENCH_ROOT))
    for line in proc.stdout.split("\n"):
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    return {"success": False, "output_dir": None, "final_turn": 0,
            "error": f"no RESULT_JSON. stderr tail: {proc.stderr[-800:]!r}"}


def serialize_rollout(output_dir: str, max_turn: int) -> dict | None:
    """Serialize a rollout recording dir into game_data via MetricsCollector.

    IMPORTANT: MetricsCollector returns dicts with int keys (turn/player),
    but QuestionResolver expects str keys (it normally reads game_data from
    JSON). We round-trip through write_world_data -> json.load so the keys
    match exactly what the production question pipeline resolves against.
    """
    import tempfile, os
    from civrealm.world_reports import ReportConfig
    from civrealm.world_reports.data_loader import DataLoader
    from civrealm.world_reports.extractors import MetricsCollector, write_world_data

    dl = DataLoader(output_dir)
    summ = dl.get_turn_summary()
    lo, hi = summ["turn_range"]
    states = dl.get_states_range(lo, hi)
    if not states:
        return None
    config = ReportConfig(recording_dir=output_dir, output_dir=output_dir,
                          report_turns=[hi], enabled_sections=["overview"], formats=[])
    collector = MetricsCollector()
    data = collector.collect_all(states=states, config=config, data_loader=dl)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        write_world_data(data, tmp)
        with open(tmp) as f:
            return json.load(f)
    finally:
        os.unlink(tmp)


def resolve_questions(questions_path: str, game_data: dict,
                      resolution_turns: list[int]) -> dict[str, bool]:
    """Resolve questions whose resolution_turn is in resolution_turns.
    Returns {question_id: bool answer}."""
    from civrealm.world_reports.questions import load_question_bank, QuestionResolver

    bank = load_question_bank(questions_path)
    resolver = QuestionResolver()
    out = {}
    rt_set = set(resolution_turns)
    for q in bank.questions:
        if q.resolution_turn not in rt_set:
            continue
        try:
            res = resolver.resolve(q, game_data, bank.snapshot_turn)
            out[q.question_id] = bool(res.answer)
        except Exception as e:
            out[q.question_id] = None  # unresolvable for this rollout
    return out


def main() -> int:
    args = _pre_args()
    end_turn = max(args.resolution_turns)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"== MC resolve: {args.game_id} snapshot={args.snapshot_turn} "
          f"-> {args.resolution_turns}  N={len(args.rng_seeds)} ==")

    # answers[question_id] -> list of bools across rollouts
    answers: dict[str, list] = {}
    rollout_log = []
    baseline_answers = None
    t0 = time.time()

    jobs = list(args.rng_seeds)
    if args.baseline:
        jobs = [None] + jobs  # None == baseline

    for i, rng_seed in enumerate(jobs):
        is_base = rng_seed is None
        tag = "base" if is_base else f"s{rng_seed}"
        fork_name = f"mcr{args.snapshot_turn}{tag}"
        ts = time.time()
        r = run_rollout_subprocess(args.game_id, args.base_seed, args.snapshot_turn,
                                   end_turn, rng_seed, fork_name)
        if not r["success"]:
            print(f"  [{i+1}/{len(jobs)}] {tag}: FORK FAIL ({(r.get('error') or '')[:200]})")
            rollout_log.append({"tag": tag, "ok": False, "error": r.get("error")})
            out_path.write_text(json.dumps({"config": vars(args), "answers": answers,
                                            "baseline": baseline_answers,
                                            "rollout_log": rollout_log}, indent=2))
            continue
        gd = serialize_rollout(r["output_dir"], end_turn)
        if gd is None:
            print(f"  [{i+1}/{len(jobs)}] {tag}: SERIALIZE FAIL")
            rollout_log.append({"tag": tag, "ok": False, "error": "serialize failed"})
            continue
        resolved = resolve_questions(args.questions, gd, args.resolution_turns)
        n_true = sum(1 for v in resolved.values() if v is True)
        if is_base:
            baseline_answers = resolved
        else:
            for qid, ans in resolved.items():
                answers.setdefault(qid, []).append(ans)
        dt = time.time() - ts
        print(f"  [{i+1}/{len(jobs)}] {tag}: ok final_turn={r['final_turn']} "
              f"resolved={len(resolved)} true={n_true} ({dt:.0f}s)")
        rollout_log.append({"tag": tag, "ok": True, "final_turn": r["final_turn"],
                            "n_resolved": len(resolved), "n_true": n_true})

        # persist partial
        out_path.write_text(json.dumps({"config": vars(args), "answers": answers,
                                        "baseline": baseline_answers,
                                        "rollout_log": rollout_log,
                                        "elapsed_sec": time.time() - t0}, indent=2))

    # Aggregate p_mc
    p_mc = {}
    for qid, vals in answers.items():
        clean = [v for v in vals if v is not None]
        if clean:
            p_mc[qid] = sum(1 for v in clean if v) / len(clean)
    out_path.write_text(json.dumps({"config": vars(args), "answers": answers,
                                    "baseline": baseline_answers, "p_mc": p_mc,
                                    "n_rollouts": {q: len([v for v in vs if v is not None])
                                                   for q, vs in answers.items()},
                                    "rollout_log": rollout_log,
                                    "elapsed_sec": time.time() - t0}, indent=2))
    print(f"Done in {time.time()-t0:.0f}s. {len(p_mc)} questions resolved. Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
