#!/usr/bin/env -S uv run python3
"""Ground truth for the binary and continuous evals from reseeded continuations.

For every (city, disasters, snapshot turn) in the config, runs the engine's
run_continuations.js — the trunk to the snapshot, then `branch_nseeds`
continuations from that exact state, each with its own RNG seed — and records
which continuations resolve each binary question Yes at every horizon (counts
and yes_idx), along with each continuous metric's value at the resolution turn. Writes one
JSONL file per (scenario, snapshot) to data/micropolis/ground_truth/, one line
per horizon.

Continuations are sharded across --jobs producer processes with disjoint seed
ranges. A (scenario, snapshot) whose file already covers every horizon with at
least branch_nseeds continuations is skipped unless --force-regen is given.
Horizon 0 is a read-off, not a forecast, and is dropped.

--intervene ACTION (repeatable) applies the intervention once at the snapshot
turn, before branching, so all continuations share its outcome, and writes
<scenario>_T<S>_do-<ACTION>.jsonl instead. A scenario where it has no effect
(e.g. makeFlood without river edges) fails, or with --skip-infeasible is
recorded in <output>.infeasible.json and skipped.

Usage:
    scripts/extract_ground_truth.py                    # packaged example-binary.json5
    scripts/extract_ground_truth.py configs/example-binary.json5 --branch-nseeds 1
    scripts/extract_ground_truth.py --cities kyoto --jobs 4 --force-regen
    scripts/extract_ground_truth.py --cities kobe --intervene demolishPowerPlants
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

from micropolis_world.binary_questions import check_horizons
from micropolis_world.city_sim import INTERVENTION_ACTIONS
from micropolis_world.config import (
    CONFIG_DIR,
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.continuous_eval import engine_commit_note, git_commit_note
from micropolis_world.ground_truth import (
    InfeasibleIntervention,
    covers,
    cross_check_trunk,
    ground_truth_lines,
    infeasible_marker,
    load_lines,
    merge_shards,
    output_path,
    run_shard,
    seed_shards,
    write_lines,
)
from micropolis_world.scenarios import get_base_scenarios

DEFAULT_CONFIG_PATH = CONFIG_DIR / "example-binary.json5"


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default=DEFAULT_CONFIG_PATH)
    ap.add_argument(
        "--branch-nseeds",
        type=int,
        default=None,
        help="Continuations per (scenario, snapshot), overriding the config's "
        "'branch_nseeds'",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Producer processes to shard the continuations over (default: CPU count)",
    )
    ap.add_argument(
        "--force-regen",
        action="store_true",
        help="Re-run scenarios whose output file already covers every horizon",
    )
    ap.add_argument(
        "--intervene",
        action="append",
        default=[],
        choices=INTERVENTION_ACTIONS,
        metavar="ACTION",
        help="Intervention applied once at the snapshot turn, before branching "
        f"(repeatable): {', '.join(INTERVENTION_ACTIONS)}",
    )
    ap.add_argument(
        "--skip-infeasible",
        action="store_true",
        help="With --intervene: record a scenario where the intervention has no "
        "effect in <output>.infeasible.json and go on, instead of failing; "
        "recorded ones are skipped on later runs unless --force-regen",
    )
    args = ap.parse_args()
    interventions = args.intervene
    if args.skip_infeasible and not interventions:
        ap.error("--skip-infeasible needs --intervene")

    cfg = load_config(args)
    seed = cfg.get_seed(args.seed)
    nseeds = (
        args.branch_nseeds
        if args.branch_nseeds is not None
        else cfg.get_int("branch_nseeds")
    )
    if nseeds < 1 or args.jobs < 1:
        ap.error("--branch-nseeds and --jobs must be positive")
    snapshot_turns = cfg.get_int_list("snapshot_turns")
    horizons = sorted(h for h in cfg.get_int_list("horizons") if h > 0)
    if not horizons:
        ap.error("the config has no positive horizon")
    try:
        check_horizons(horizons)
    except ValueError as e:
        ap.error(str(e))
    scenarios = get_base_scenarios(
        seed=seed,
        cities=cfg.get_cities(args.cities),
        disasters=cfg.get_disasters(args.disasters),
    )
    fbsim_commit = git_commit_note()
    engine_commit = engine_commit_note()

    print("=" * 70)
    print("MICROPOLIS WORLD — ground truth from continuations")
    print("=" * 70)
    print(f"config: {cfg.path}")
    print(
        f"{len(scenarios)} scenario(s) x {len(snapshot_turns)} snapshot(s), "
        f"{nseeds} continuations each over {args.jobs} job(s); horizons {horizons}"
    )
    if interventions:
        print(f"interventions at the snapshot turn: {', '.join(interventions)}")

    jobs = [(sim, S) for sim in scenarios for S in snapshot_turns]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, (sim, S) in enumerate(jobs, start=1):
            path = output_path(sim, S, interventions)
            tag = f"[{i}/{len(jobs)}] {sim.get_id_str()} T{S}"
            if (
                not args.force_regen
                and path.exists()
                and covers(load_lines(path), horizons, nseeds)
            ):
                print(f"{tag}: already covered by {path}, skipping")
                continue
            marker = infeasible_marker(path)
            if args.skip_infeasible and not args.force_regen and marker.exists():
                print(f"{tag}: recorded as infeasible in {marker}, skipping")
                continue
            print(f"{tag}: running {nseeds} continuations to T{S + max(horizons)}...")
            start = time.perf_counter()
            futures = [
                pool.submit(
                    run_shard,
                    sim.city_name,
                    seed,
                    sim.disasters,
                    S,
                    horizons,
                    lo,
                    hi,
                    interventions,
                )
                for lo, hi in seed_shards(nseeds, args.jobs)
            ]
            try:
                merged = merge_shards([f.result() for f in futures])
            except InfeasibleIntervention as e:
                for f in futures:
                    f.cancel()
                if not args.skip_infeasible:
                    raise
                record = {
                    "scenario_id": sim.get_id_str(),
                    "snapshot_turn": S,
                    "interventions": interventions,
                    "error": str(e),
                    "engine_commit": engine_commit,
                }
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps(record) + "\n")
                print(f"  infeasible ({e}); recorded in {marker}")
                continue
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
            print(f"  {cross_check_trunk(merged.trunk_log, sim)}")
            write_lines(
                path,
                ground_truth_lines(
                    sim, S, horizons, merged, fbsim_commit, engine_commit, interventions
                ),
            )
            print(
                f"  {merged.n} continuations in {time.perf_counter() - start:.0f}s -> {path}"
            )
    print("=" * 70)


if __name__ == "__main__":
    main()
