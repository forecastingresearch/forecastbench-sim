"""Track A orchestrator: run the free-play A/B in parallel subprocesses.

Spawns agentic_run.py for each (seed, arm, repeat), limited concurrency (each
game itself issues several concurrent Gemini calls, and freeciv has ~8 ports).
Collects final gold and writes a combined results file.

Usage:
  PYTHONPATH=src:llm_baseline uv run python scripts/uplift/agentic_ab.py \
    --seeds 100 101 102 103 104 --arms control forecast --repeats 2 \
    --horizon 28 --forecasts data/uplift/forecasts_agentic.json \
    --concurrency 4 --outdir tmp/uplift/agentic --output data/uplift/agentic_ab.json
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def run_one(seed, arm, rep, horizon, ai, forecasts, model, outdir):
    out = Path(outdir) / f"ag_s{seed}_{arm}_r{rep}.json"
    cmd = [sys.executable, str(ROOT / "scripts" / "uplift" / "agentic_run.py"),
           "--seed", str(seed), "--arm", arm, "--horizon", str(horizon),
           "--ai", str(ai), "--output", str(out)]
    if forecasts:
        cmd += ["--forecasts", forecasts]
    if model:
        cmd += ["--model", model]
    env = {"PYTHONPATH": f"{ROOT}/src:{ROOT}/llm_baseline"}
    import os
    full_env = dict(os.environ); full_env.update(env)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                              env=full_env, timeout=horizon * 300 + 1800)
    except subprocess.TimeoutExpired:
        return {"seed": seed, "arm": arm, "rep": rep, "final_gold": None,
                "error": "timeout"}
    if out.exists():
        r = json.loads(out.read_text())
        r["rep"] = rep
        print(f"  done s{seed} {arm} r{rep}: gold={r.get('final_gold')} "
              f"turn={r.get('final_turn')} ({time.time()-t0:.0f}s)", flush=True)
        return r
    return {"seed": seed, "arm": arm, "rep": rep, "final_gold": None,
            "error": f"no output. stderr tail: {proc.stderr[-400:]!r}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=["control", "forecast"])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--ai", type=int, default=5)
    ap.add_argument("--forecasts", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--outdir", default="tmp/uplift/agentic")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    jobs = [(s, a, r) for s in args.seeds for a in args.arms
            for r in range(args.repeats)]
    print(f"Running {len(jobs)} games (concurrency={args.concurrency}) ...")
    results = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, s, a, r, args.horizon, args.ai,
                          args.forecasts, args.model, args.outdir): i
                for i, (s, a, r) in enumerate(jobs)}
        for fut, i in futs.items():
            results[i] = fut.result()
            Path(args.output).write_text(json.dumps(
                {"config": vars(args), "games": [x for x in results if x]}, indent=2))
    print(f"Wrote {len([r for r in results if r])} games -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
