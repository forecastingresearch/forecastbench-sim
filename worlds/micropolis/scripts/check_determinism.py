#!/usr/bin/env -S uv run python3
"""Check whether the Micropolis engine is deterministic given a fixed seed.

Runs the same scenario several times at the same seed and compares the log and
events files byte-for-byte across runs, then field-by-field to say exactly where
any two runs first diverge.

CitySimulation.run writes to a path fixed by (city, seed, disasters), so each
repeat overwrites the last. Every repeat's output is copied into a scratch
directory before the next one starts, and the originals are restored afterwards
so a check leaves data/micropolis/runs/ as it found it.

--plot additionally writes one figure per scenario overlaying every repeat's
metrics on the same panels plot_sim.py uses, so the run-to-run spread is visible
rather than only tabulated. Disaster scenarios are skipped there.

Usage:
    scripts/check_determinism.py
    scripts/check_determinism.py my_config.json5 --repeats 5 --plot
    scripts/check_determinism.py --repeats 3 --turns 200 --keep
    scripts/check_determinism.py --cities kyoto --disasters false --seed 7
"""

import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from micropolis_world import module_globals as g
from micropolis_world.city_sim import CitySimulation
from micropolis_world.config import (
    add_config_args,
    load_config,
    main_with_config,
    scenarios_from,
)

# The two per-run outputs a scenario produces. The world reports and the plot
# are derived from the log, so a log that matches implies they do too.
RUN_FILES = ["log", "events"]

# Enough turns for divergence to compound and show up, without paying for a
# full 1000-turn run on every repeat. Override with --turns.
DEFAULT_TURNS = 300

# Enough histories to see the spread in a --plot figure without the panels
# turning into a thicket. Also the number compared when only reporting.
DEFAULT_REPEATS = 5


# --plot figures go beside the run data rather than into data/micropolis/runs/,
# whose per-city plot filenames belong to run_sim.py's single-run figures.
def plots_dir() -> Path:
    """Under the process's data directory, which the config fixes after import."""
    return g.DATA_DIR / "determinism"


def snapshot_run(sim: CitySimulation, dest: Path) -> dict[str, Path]:
    """Copy this run's output files into `dest`, returning the copies by prefix.

    The engine writes each run to the same path, so the files have to be moved
    aside before the next repeat overwrites them.
    """
    dest.mkdir(parents=True, exist_ok=True)
    copies = {}
    for prefix in RUN_FILES:
        source = sim.get_data_file_path(prefix)
        if not source.exists():
            raise FileNotFoundError(f"expected simulation output at {source}")
        copy = dest / source.name
        shutil.copy2(source, copy)
        copies[prefix] = copy
    return copies


def load_rows(path: Path) -> list[dict]:
    """The JSONL rows of a log or events file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def first_row_difference(
    rows_a: list[dict], rows_b: list[dict]
) -> tuple[int, str, object, object] | None:
    """The first (row index, field, value_a, value_b) where two runs disagree.

    Fields are compared in the first row's key order, so the field named is the
    leftmost differing one in the log — which tends to be the closest thing to a
    cause. A row count mismatch is reported as a difference in the row that only
    one run has.
    """
    for i in range(min(len(rows_a), len(rows_b))):
        a, b = rows_a[i], rows_b[i]
        for key in list(a) + [k for k in b if k not in a]:
            if a.get(key) != b.get(key):
                return i, key, a.get(key), b.get(key)
    if len(rows_a) != len(rows_b):
        return min(len(rows_a), len(rows_b)), "<row count>", len(rows_a), len(rows_b)
    return None


def differing_fields(rows_a: list[dict], rows_b: list[dict]) -> Counter:
    """How many rows each field differs in, across the rows both runs have."""
    counts: Counter = Counter()
    for a, b in zip(rows_a, rows_b):
        for key in list(a) + [k for k in b if k not in a]:
            if a.get(key) != b.get(key):
                counts[key] += 1
    return counts


def compare_scenario(
    sim: CitySimulation, runs: list[dict[str, Path]], quiet: bool
) -> bool:
    """Report on one scenario's repeats; True if every repeat matched run 1.

    Bytes are the real test: identical files mean identical time series and
    identical everything else the run wrote. A byte difference is then read back
    as JSON to name the turn and field it starts at, since "the files differ" on
    its own says nothing about whether the divergence is cosmetic or dynamical.
    """
    identical = True
    for prefix in RUN_FILES:
        baseline = runs[0][prefix]
        baseline_bytes = baseline.read_bytes()
        mismatched = [
            i
            for i, run in enumerate(runs[1:], start=2)
            if run[prefix].read_bytes() != baseline_bytes
        ]
        if not mismatched:
            if not quiet:
                rows = len(load_rows(baseline))
                print(
                    f"  {prefix}: identical across all {len(runs)} runs ({rows} rows)"
                )
            continue

        identical = False
        runs_word = "run" if len(mismatched) == 1 else "runs"
        print(
            f"  {prefix}: DIFFERS — {len(mismatched)} of {len(runs) - 1} repeats "
            f"do not match run 1 ({runs_word} {', '.join(map(str, mismatched))})"
        )
        baseline_rows = load_rows(baseline)
        for i in mismatched:
            other_rows = load_rows(runs[i - 1][prefix])
            diff = first_row_difference(baseline_rows, other_rows)
            if diff is None:
                # Byte-different but JSON-equal: key order or formatting only.
                print(
                    f"    run {i}: bytes differ but parsed rows are equal "
                    "(formatting only)"
                )
                continue
            index, field, value_a, value_b = diff
            turn = f", turn {index}" if prefix == "log" else ""
            print(
                f"    run {i}: first difference at row {index}{turn}, "
                f"field {field!r}: {value_a!r} vs {value_b!r}"
            )
            fields = differing_fields(baseline_rows, other_rows)
            top = ", ".join(f"{k} ({n})" for k, n in fields.most_common(6))
            print(
                f"      {len(fields)} field(s) differ in "
                f"{sum(1 for a, b in zip(baseline_rows, other_rows) if a != b)} row(s)"
                f"; most affected: {top}"
            )
    return identical


def plot_scenario_repeats(
    sim: CitySimulation, runs: list[dict[str, Path]], repeats: int
) -> None:
    """Write one figure overlaying every repeat's metrics for this scenario.

    Reads the snapshot copies, not sim.log_data: the originals are restored on
    the way out, and only run `repeats` is still on disk under its real path.

    Disaster runs are skipped rather than drawn without their disaster lines,
    which would show a spread that mixes RNG divergence with the effect of a
    strike hitting one run and not another.
    """
    if sim.disasters:
        print("  plot: skipped (disasters enabled)")
        return

    # Imported here so a check without --plot doesn't pull in matplotlib.
    from micropolis_world.plot_sim import plot_repeats

    plots_dir().mkdir(parents=True, exist_ok=True)
    output = plots_dir() / f"repeats-{sim.get_id_str()}-x{repeats}.png"
    plot_repeats(
        [load_rows(run["log"]) for run in runs],
        title=f"{sim.get_id_str()} — {repeats} runs at the same seed",
        output=str(output),
    )


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"How many times to run each scenario (default: {DEFAULT_REPEATS})",
    )
    ap.add_argument(
        "--turns",
        type=int,
        default=DEFAULT_TURNS,
        help=f"Turns per run (default: {DEFAULT_TURNS})",
    )
    ap.add_argument(
        "--keep",
        action="store_true",
        help="Keep the per-run output copies instead of deleting them",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help=(
            "Also write a figure per scenario overlaying every repeat's metrics, "
            f"into {plots_dir()}. Skips scenarios with disasters enabled"
        ),
    )
    ap.add_argument("--quiet", action="store_true", help="Only report differences")
    args = ap.parse_args()

    if args.repeats < 2:
        ap.error("--repeats must be at least 2 to have anything to compare")

    cfg = load_config(args)
    seed = cfg.get_seed(args.seed)
    scenarios = scenarios_from(cfg, args.cities, args.disasters)

    workdir = Path(tempfile.mkdtemp(prefix="micropolis-determinism-"))
    print("=" * 70)
    print("MICROPOLIS DETERMINISM CHECK")
    print("=" * 70)
    print(
        f"{len(scenarios)} scenario(s) x {args.repeats} repeats at seed {seed}, "
        f"{args.turns} turns each"
    )
    print(f"per-run copies: {workdir}")

    # The check overwrites the cached runs in data/micropolis/runs/, so put back
    # whatever was there — the other scripts reuse those files, and a shorter
    # --turns run left behind would silently truncate their corpus.
    backup = workdir / "_restore"
    restore: list[tuple[Path, Path]] = []
    for city, disasters in scenarios:
        sim = CitySimulation(city_name=city, seed=seed, disasters=disasters)
        for prefix in RUN_FILES:
            existing = sim.get_data_file_path(prefix)
            if existing.exists():
                saved = backup / sim.get_id_str() / existing.name
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(existing, saved)
                restore.append((saved, existing))

    nondeterministic = []
    try:
        for city, disasters in scenarios:
            sim = CitySimulation(city_name=city, seed=seed, disasters=disasters)
            print(f"\n{sim.get_id_str()}")
            runs = []
            for i in range(1, args.repeats + 1):
                if not args.quiet:
                    print(f"  run {i}/{args.repeats}...")
                sim.run(nturns=args.turns, quiet=True)
                runs.append(snapshot_run(sim, workdir / sim.get_id_str() / f"run{i}"))
            if not compare_scenario(sim, runs, args.quiet):
                nondeterministic.append(sim.get_id_str())
            if args.plot:
                plot_scenario_repeats(sim, runs, args.repeats)
    finally:
        for saved, original in restore:
            shutil.copy2(saved, original)
        if restore:
            print(f"\nrestored {len(restore)} cached run file(s)")
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"kept per-run copies in {workdir}")

    print("\n" + "=" * 70)
    if nondeterministic:
        print(
            f"NON-DETERMINISTIC: {len(nondeterministic)} of {len(scenarios)} "
            "scenario(s) did not reproduce at a fixed seed"
        )
        for scenario_id in nondeterministic:
            print(f"  {scenario_id}")
    else:
        print(
            f"DETERMINISTIC: all {len(scenarios)} scenario(s) reproduced exactly "
            f"over {args.repeats} runs at seed {seed}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
