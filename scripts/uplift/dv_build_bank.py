"""Track B scenario-bank builder (offline simulation).

For each scenario (seed, checkpoint turn T) this computes, entirely with the
built-in FreeCiv AI (no LLM):

  1. Game-true forecasts p_mc for a set of events, from N status-quo rollouts
     (reseeded RNG, no policy change) resolved at two horizons.
  2. The ground-truth gold outcome of each candidate policy (government choice),
     from N reseeded rollouts per policy -> mean gold at T+H.

The LLM decision experiment (dv_decide.py) then reuses this bank: it is cheap
(LLM only chooses a policy) and its outcome = the precomputed gold of the chosen
policy. This amortizes the expensive simulation and is robust to *how* a policy
takes effect (e.g. the AI may revolt after a forced government change; that
response is baked consistently into every arm's ground truth).

Usage:
  PYTHONPATH=src uv run python scripts/uplift/dv_build_bank.py \
    --seeds 0 --checkpoints 40 48 56 64 --horizon 18 --n 8 --workers 6 \
    --output data/uplift/bank_pilot.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CIVBENCH_ROOT, run_rollouts_parallel, find_state_json,
                    parse_state_json, parse_savegame, final_gold)

GOVERNMENTS = ["Despotism", "Monarchy", "Republic", "Democracy"]

# Each event: name -> (predicate on resolved-state dict, given checkpoint baseline).
# Kept to events that (a) bear on the government/gold trade-off and (b) resolve
# robustly from recorded states. None of them is the gold answer itself.
def event_values(ev: dict, base: dict) -> dict:
    return {
        "at_war": bool(ev.get("at_war")),
        "lose_city": (ev.get("n_cities") is not None and base.get("n_cities") is not None
                      and ev["n_cities"] < base["n_cities"]),
        "gain_city": (ev.get("n_cities") is not None and base.get("n_cities") is not None
                      and ev["n_cities"] > base["n_cities"]),
        "army_shrinks": (ev.get("n_units") is not None and base.get("n_units") is not None
                         and ev["n_units"] < base["n_units"]),
        "rival_ahead": (ev.get("rival_max_score") is not None and ev.get("score") is not None
                        and ev["rival_max_score"] > ev["score"]),
        "top_score": bool(ev.get("top_score")),
    }

EVENT_NAMES = ["at_war", "lose_city", "gain_city", "army_shrinks", "rival_ahead", "top_score"]


def checkpoint_baseline(seed: int, ckpt: int) -> dict:
    """City/unit counts (and gold/gov) at the checkpoint, for event deltas."""
    rec = CIVBENCH_ROOT / "logs" / "recordings" / f"seed{seed}" / "savegames"
    cands = sorted(rec.glob(f"seed{seed}_T{ckpt}_*.sav.xz"))
    if not cands:
        raise FileNotFoundError(f"no checkpoint savegame for seed{seed} T{ckpt}")
    return parse_savegame(str(cands[0]), 0)


def build_scenario(seed: int, ckpt: int, horizon: int, fc_horizons: list,
                   n: int, workers: int, governments: list = GOVERNMENTS) -> dict:
    rec_dir = f"logs/recordings/seed{seed}"
    end_turn = ckpt + horizon
    base = checkpoint_baseline(seed, ckpt)

    # Build all rollout specs: status-quo (always, for forecasts) + one set per
    # government policy (empty governments => forecast-only, ~5x faster).
    policies = ["statusquo"] + list(governments)
    specs, meta = [], []
    for pol in policies:
        extra = None if pol == "statusquo" else [
            {"type": "government", "player_id": 0, "value": pol}]
        for s in range(1, n + 1):
            fn = f"dv{seed}c{ckpt}{'sq' if pol=='statusquo' else pol[:3]}s{s}"
            specs.append(dict(recording_dir=rec_dir, base_seed=seed,
                              checkpoint_turn=ckpt, end_turn=end_turn,
                              rng_seed=s, fork_name=fn, extra_mods=extra))
            meta.append((pol, s))

    t0 = time.time()
    results = run_rollouts_parallel(specs, max_workers=workers)
    elapsed = time.time() - t0

    # Resolve everything, then delete fork output dirs (disk hygiene: each rollout
    # writes per-turn state JSONs; hundreds of rollouts would otherwise pile up GBs).
    import shutil
    fork_dirs = {r["output_dir"] for r in results if r.get("output_dir")}

    # Collect per-policy gold at horizon, and status-quo event resolutions.
    gold_by_policy = {p: [] for p in policies}
    fc_hits = {h: {e: [] for e in EVENT_NAMES} for h in fc_horizons}
    sq_gold_traj = []
    for (pol, s), r in zip(meta, results):
        if not r.get("success"):
            continue
        g = final_gold(r, 0)
        if g is not None:
            gold_by_policy[pol].append(g)
        if pol == "statusquo" and r.get("output_dir"):
            for h in fc_horizons:
                sj = find_state_json(r["output_dir"], ckpt + h)
                if not sj:
                    continue
                ev = parse_state_json(sj, 0)
                vals = event_values(ev, base)
                for e in EVENT_NAMES:
                    fc_hits[h][e].append(1 if vals[e] else 0)

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    gold_stats = {p: {"mean": mean(v), "n": len(v),
                      "vals": sorted(round(x, 1) for x in v)}
                  for p, v in gold_by_policy.items()}
    govs = {g: gold_stats[g]["mean"] for g in governments
            if gold_stats.get(g, {}).get("mean") is not None}
    best_gov = max(govs, key=govs.get) if govs else None

    p_mc = {f"h{h}": {e: (mean(fc_hits[h][e]) if fc_hits[h][e] else None)
                       for e in EVENT_NAMES} for h in fc_horizons}
    n_mc = {f"h{h}": len(fc_hits[h][EVENT_NAMES[0]]) for h in fc_horizons}

    for d in fork_dirs:
        try:
            shutil.rmtree(d)
        except Exception:  # noqa: BLE001
            pass

    return {
        "seed": seed, "checkpoint": ckpt, "horizon": horizon,
        "end_turn": end_turn, "fc_horizons": fc_horizons, "n_rollouts": n,
        "baseline": {k: base.get(k) for k in ("gold", "government", "n_cities", "n_units")},
        "gold_by_policy": gold_stats,
        "best_gov": best_gov,
        "gold_spread": (max(govs.values()) - min(govs.values())) if govs else None,
        "p_mc": p_mc, "n_mc": n_mc,
        "elapsed_sec": round(elapsed, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--checkpoints", type=int, nargs="+", required=True)
    ap.add_argument("--horizon", type=int, default=18)
    ap.add_argument("--fc-horizons", type=int, nargs="+", default=None,
                    help="forecast horizons relative to checkpoint (default: H/2, H)")
    ap.add_argument("--n", type=int, default=8, help="rollouts per policy")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--governments", nargs="*", default=GOVERNMENTS,
                    help="government policies to score (empty => forecast-only)")
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    fc_h = args.fc_horizons or [max(1, args.horizon // 2), args.horizon]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scenarios = []
    # Resume support: keep already-built (seed, ckpt) if output exists.
    done = set()
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        scenarios = prev.get("scenarios", [])
        done = {(s["seed"], s["checkpoint"]) for s in scenarios}

    todo = [(sd, ck) for sd in args.seeds for ck in args.checkpoints
            if (sd, ck) not in done]
    print(f"Building {len(todo)} scenarios (skipping {len(done)} done); "
          f"H={args.horizon} fc_h={fc_h} n={args.n} workers={args.workers}")
    for i, (sd, ck) in enumerate(todo):
        t0 = time.time()
        try:
            sc = build_scenario(sd, ck, args.horizon, fc_h, args.n, args.workers,
                                governments=args.governments)
        except Exception as e:  # noqa: BLE001
            print(f"[{i+1}/{len(todo)}] seed{sd} T{ck} FAILED: {e!r}")
            continue
        scenarios.append(sc)
        gp = {g: (round(v['mean'], 0) if v.get('mean') is not None else None)
              for g, v in sc['gold_by_policy'].items() if g != 'statusquo'}
        print(f"[{i+1}/{len(todo)}] seed{sd} T{ck} ({time.time()-t0:.0f}s) "
              f"best={sc['best_gov']} spread={sc['gold_spread']} gold={gp}")
        print(f"      p_mc h{fc_h[-1]}: "
              + ", ".join(f"{e}={sc['p_mc'][f'h{fc_h[-1]}'][e]}" for e in EVENT_NAMES))
        out_path.write_text(json.dumps(
            {"config": vars(args), "governments": GOVERNMENTS,
             "event_names": EVENT_NAMES, "scenarios": scenarios}, indent=2))
    print(f"Wrote {len(scenarios)} scenarios -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
