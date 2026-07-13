#!/usr/bin/env python3
"""Causal golden-question pilot: MC rollouts that keep full per-rollout game data.

Extends mc_resolve.py for the conditioning-vs-intervening study. For each rollout:
  1. fork the snapshot savegame (RNG reseed + optional intervention mods),
  2. run to the max resolution turn,
  3. serialize the rollout to game_data and save it gzipped (post-hoc event
     conditioning: tech discovery turns, per-turn series, government changes),
  4. resolve the game's question bank against the rollout,
  5. delete the fork recording dir (each is ~70 MB; disk is tight).

Arms:
  baseline:  no intervention -> estimates P(Y) and, by filtering rollouts where
             an event X happened naturally, P(Y | X observed).
  do(X):     pass --intervention (e.g. "tech:0:23") -> estimates P(Y | do(X)).

Usage:
    PYTHONPATH=src uv run python scripts/causal_pilot.py \
        --game-id seed0 --base-seed 0 --snapshot-turn 60 --resolution-turns 90 \
        --rng-seeds $(seq 1001 1040) \
        --questions data/questions_mc/seed0/questions.json \
        --outdir tmp/causal/seed0/baseline

    # do(X) arm:
    ... --intervention tech:0:23 --outdir tmp/causal/seed0/do_tech23p0
"""
from __future__ import annotations

import sys
import json
import gzip
import shutil
import argparse
import subprocess
import time
from pathlib import Path

sys.path.insert(0, "src")

CIVBENCH_ROOT = Path(__file__).parent.parent


def parse_intervention(spec: str) -> dict:
    """'tech:0:23' -> {type: tech, player_id: 0, tech_id: 23};
    'gold_add:0:500' / 'gold:0:5000' / 'government:1:Republic' likewise."""
    parts = spec.split(":")
    kind, player_id = parts[0], int(parts[1])
    if kind == "tech":
        return {"type": "tech", "player_id": player_id, "tech_id": int(parts[2])}
    if kind in ("gold", "gold_add"):
        return {"type": kind, "player_id": player_id, "value": int(parts[2])}
    if kind == "government":
        return {"type": "government", "player_id": player_id, "value": parts[2]}
    raise ValueError(f"unknown intervention spec: {spec}")


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--game-id", required=True)
    p.add_argument("--base-seed", type=int, required=True)
    p.add_argument("--snapshot-turn", type=int, required=True)
    p.add_argument("--resolution-turns", type=int, nargs="+", required=True)
    p.add_argument("--rng-seeds", type=int, nargs="+", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--intervention", action="append", default=[],
                   help="e.g. tech:0:23 or gold_add:0:500 (repeatable)")
    p.add_argument("--fork-prefix", default=None,
                   help="unique fork name prefix; default derived from outdir")
    return p.parse_args()


def run_rollout_subprocess(game_id: str, base_seed: int, snapshot_turn: int,
                           end_turn: int, modifications: list[dict],
                           fork_name: str) -> dict:
    recording_dir = f"logs/recordings/{game_id}"
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


def serialize_rollout(output_dir: str) -> dict | None:
    """Serialize a rollout recording dir into game_data (same as mc_resolve)."""
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
        except Exception:
            out[q.question_id] = None
    return out


def main() -> int:
    args = _args()
    end_turn = max(args.resolution_turns)
    outdir = Path(args.outdir)
    (outdir / "rollouts").mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "manifest.json"

    interventions = [parse_intervention(s) for s in args.intervention]
    prefix = args.fork_prefix or ("cp" + outdir.name.replace("_", "")[:8])

    # resume support: skip rng seeds whose rollout file already exists
    done = {p.stem.split(".")[0] for p in (outdir / "rollouts").glob("*.json.gz")}

    print(f"== causal_pilot: {args.game_id} snapshot={args.snapshot_turn} -> "
          f"{args.resolution_turns} N={len(args.rng_seeds)} "
          f"interventions={args.intervention or 'none'} ==")

    answers: dict[str, list] = {}
    rollout_log = []
    t0 = time.time()

    # reload prior answers on resume
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        answers = prior.get("answers", {})
        rollout_log = prior.get("rollout_log", [])

    for i, rng_seed in enumerate(args.rng_seeds):
        tag = f"s{rng_seed}"
        if tag in done:
            continue
        fork_name = f"{prefix}{tag}"
        mods = interventions + [{"type": "rng_seed", "value": rng_seed}]
        ts = time.time()
        r = run_rollout_subprocess(args.game_id, args.base_seed, args.snapshot_turn,
                                   end_turn, mods, fork_name)
        if not r["success"]:
            print(f"  [{i+1}/{len(args.rng_seeds)}] {tag}: FORK FAIL "
                  f"({(r.get('error') or '')[:200]})")
            rollout_log.append({"tag": tag, "ok": False, "error": r.get("error")})
        else:
            gd = serialize_rollout(r["output_dir"])
            if gd is None:
                print(f"  [{i+1}/{len(args.rng_seeds)}] {tag}: SERIALIZE FAIL")
                rollout_log.append({"tag": tag, "ok": False, "error": "serialize failed"})
            else:
                resolved = resolve_questions(args.questions, gd, args.resolution_turns)
                with gzip.open(outdir / "rollouts" / f"{tag}.json.gz", "wt") as f:
                    json.dump(gd, f)
                for qid, ans in resolved.items():
                    answers.setdefault(qid, []).append({"tag": tag, "answer": ans})
                n_true = sum(1 for v in resolved.values() if v is True)
                dt = time.time() - ts
                print(f"  [{i+1}/{len(args.rng_seeds)}] {tag}: ok "
                      f"final_turn={r['final_turn']} resolved={len(resolved)} "
                      f"true={n_true} ({dt:.0f}s)")
                rollout_log.append({"tag": tag, "ok": True,
                                    "final_turn": r["final_turn"],
                                    "n_resolved": len(resolved), "n_true": n_true})
        # delete the fork recording dir regardless of serialize outcome (disk)
        od = r.get("output_dir")
        if od and Path(od).exists():
            shutil.rmtree(od, ignore_errors=True)

        p_mc = {}
        for qid, recs in answers.items():
            clean = [x["answer"] for x in recs if x["answer"] is not None]
            if clean:
                p_mc[qid] = sum(1 for v in clean if v) / len(clean)
        manifest_path.write_text(json.dumps({
            "config": {**vars(args)}, "answers": answers, "p_mc": p_mc,
            "rollout_log": rollout_log, "elapsed_sec": time.time() - t0}, indent=2))

    n_ok = sum(1 for r in rollout_log if r.get("ok"))
    print(f"Done in {time.time()-t0:.0f}s. {n_ok}/{len(rollout_log)} rollouts ok. "
          f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
