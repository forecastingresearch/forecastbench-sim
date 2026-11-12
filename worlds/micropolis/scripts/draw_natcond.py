#!/usr/bin/env -S uv run python3
"""Draw natural-conditional cells (question A, revealed fact B) from the ground-truth replays.

Reads data/micropolis/ground_truth/ (yes_idx needed) and writes one JSONL line per
cell with P(A) and P(A|B). See micropolis_world/natcond.py for the rules.

Usage:
    scripts/draw_natcond.py                       # 400 cells, seed 42
    scripts/draw_natcond.py --n 200 --quota irrelevant=40,mechanical=80,inferential=80
"""

import argparse
import collections
import json
import statistics
from pathlib import Path

from micropolis_world import module_globals as g
from micropolis_world.ground_truth import load_lines, output_path_for
from micropolis_world.natcond import (
    BLOCKS,
    DEFAULT_MAX_GAP,
    DEFAULT_QUOTA,
    candidate_cells,
    draw,
    finish,
)


def parse_quota(text: str) -> dict[str, int]:
    quota = {k: int(v) for k, v in (kv.split("=") for kv in text.split(","))}
    if set(quota) - set(BLOCKS):
        raise argparse.ArgumentTypeError(f"blocks must be among {BLOCKS}")
    return quota


def scale(quota: dict[str, int], n: int) -> dict[str, int]:
    """Default quota proportions rescaled to n cells."""
    total = sum(quota.values())
    out = {k: v * n // total for k, v in quota.items()}
    out[BLOCKS[-1]] += n - sum(out.values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400, help="cells to draw (default 400)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    ap.add_argument("--quota", type=parse_quota, default=None,
                    help="cells per block, e.g. irrelevant=50,mechanical=200,inferential=150 "
                    "(default: those proportions scaled to --n)")
    ap.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP,
                    help="the reveal's horizon is at most this many steps before the question's "
                    f"(default {DEFAULT_MAX_GAP} = consecutive; 0 = no limit)")
    ap.add_argument("--snapshots", type=int, nargs="+", default=[960, 1440])
    ap.add_argument("--horizons", type=int, nargs="+", default=[144, 240, 336, 480])
    ap.add_argument("--cities", nargs="+", default=None,
                    help="default: every city with ground-truth files for the snapshots")
    ap.add_argument("--sim-seed", type=int, default=42, help="seed the scenarios were run from")
    ap.add_argument("--out", default=None, help="default: data/micropolis/natcond/natcond_<n>_seed<seed>.jsonl")
    args = ap.parse_args()
    quota = args.quota or scale(DEFAULT_QUOTA, args.n)
    horizons = sorted(args.horizons)

    cells, commits, used = [], set(), []
    cities = args.cities or [c for c in g.CITY_CHOICES
                             if output_path_for(f"{c}_disasters_seed{args.sim_seed}", args.snapshots[0]).exists()]
    for city in cities:
        sid = f"{city}_disasters_seed{args.sim_seed}"
        for S in args.snapshots:
            path = output_path_for(sid, S)
            if not path.exists():
                raise SystemExit(f"{path} not found; run extract_ground_truth.py for {city} T{S}")
            lines = load_lines(path)
            have = {line["horizon"] for line in lines}
            if not set(horizons) <= have or any("yes_idx" not in line for line in lines):
                raise SystemExit(f"{path} lacks horizons {sorted(set(horizons) - have)} or yes_idx; rerun extract_ground_truth.py")
            used.append(path.name)
            commits |= {line["fbsim_commit"] for line in lines}
            cells += candidate_cells(lines, horizons, args.max_gap or None)
    if not used:
        raise SystemExit("no ground-truth files found for those cities and snapshots")

    print(f"{len(used)} ground-truth files, {len(cells):,} admissible cells")
    print("  by block:", dict(collections.Counter(c["block"] for c in cells)))
    chosen, notes = draw(cells, args.n, quota, args.seed)
    for note in notes:
        print("  shortfall:", note)
    commit = ", ".join(sorted(commits))
    records = [finish(c, i, args.seed, commit) for i, c in enumerate(chosen)]

    out = Path(args.out) if args.out else g.DATA_DIR / "natcond" / f"natcond_{args.n}_seed{args.seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)

    print(f"\n{len(records)} cells -> {out}")
    tab = collections.Counter((r["snapshot_turn"], r["horizon_a"], r["block"]) for r in records)
    print("  S  H_A  " + "  ".join(f"{b:>11}" for b in BLOCKS))
    for S, h in sorted({(s, h) for s, h, _ in tab}):
        print(f"  {S:<4} {h:<4} " + "  ".join(f"{tab[S, h, b]:>11}" for b in BLOCKS))
    d = [abs(r["delta"]) for r in records]
    bins = [(0, .03), (.03, .08), (.08, .15), (.15, 1.01)]
    print("  |delta|:", ", ".join(f"[{lo},{hi}) {sum(lo <= x < hi for x in d)}" for lo, hi in bins),
          f"; median {statistics.median(d):.3f}")
    print("  cities:", dict(sorted(collections.Counter(r["city"] for r in records).items())))


if __name__ == "__main__":
    main()
