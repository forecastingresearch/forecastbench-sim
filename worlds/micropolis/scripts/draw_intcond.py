#!/usr/bin/env -S uv run python3
"""Draw interventional-conditional cells (question A, intervention X at the snapshot).

Reads the base and do-X ground truth in data/micropolis/ground_truth/ (yes_idx
needed) and writes one JSONL line per cell with P(A) and P(A|do X). A
(scenario, snapshot, X) without a do-file (not extracted, or recorded as
infeasible) contributes no cells. See micropolis_world/intcond.py for the rules.

Usage:
    scripts/draw_intcond.py                       # 400 cells, seed 42
    scripts/draw_intcond.py --n 200 --quota null=25,same-family=25,damage=100,infrastructure=50
"""

import argparse
import collections
import json
import statistics
from pathlib import Path

from micropolis_world import module_globals as g
from micropolis_world.ground_truth import load_lines, output_path_for
from micropolis_world.intcond import (
    BLOCKS,
    DEFAULT_QUOTA,
    X_FAMILY,
    candidate_cells,
    draw,
    finish,
    weak_pairs,
)
from micropolis_world.natcond import FAMILY


def parse_quota(text: str) -> dict[str, int]:
    quota = {k: int(v) for k, v in (kv.split("=") for kv in text.split(","))}
    if set(quota) - set(BLOCKS):
        raise argparse.ArgumentTypeError(f"blocks must be among {BLOCKS}")
    return quota


def scale(quota: dict[str, int], n: int) -> dict[str, int]:
    """Default quota proportions rescaled to n cells."""
    total = sum(quota.values())
    out = {k: v * n // total for k, v in quota.items()}
    out["damage"] += n - sum(out.values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400, help="cells to draw (default 400)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    ap.add_argument("--quota", type=parse_quota, default=None,
                    help="cells per block, e.g. null=50,same-family=50,damage=200,infrastructure=100 "
                    "(default: those proportions scaled to --n)")
    ap.add_argument("--actions", nargs="+", default=list(X_FAMILY), choices=list(X_FAMILY))
    ap.add_argument("--snapshots", type=int, nargs="+", default=[960, 1440])
    ap.add_argument("--horizons", type=int, nargs="+", default=[144, 240, 336, 480])
    ap.add_argument("--cities", nargs="+", default=None,
                    help="default: every city with base ground truth for the snapshots")
    ap.add_argument("--sim-seed", type=int, default=42, help="seed the scenarios were run from")
    ap.add_argument("--out", default=None, help="default: data/micropolis/intcond/intcond_<n>_seed<seed>.jsonl")
    args = ap.parse_args()
    quota = args.quota or scale(DEFAULT_QUOTA, args.n)
    horizons = sorted(args.horizons)

    def checked(path: Path) -> list[dict]:
        lines = load_lines(path)
        have = {line["horizon"] for line in lines}
        if not set(horizons) <= have or any("yes_idx" not in line for line in lines):
            raise SystemExit(f"{path} lacks horizons {sorted(set(horizons) - have)} or yes_idx; rerun extract_ground_truth.py")
        return lines

    cells, commits, used, missing = [], set(), [], []
    cities = args.cities or [c for c in g.CITY_CHOICES
                             if output_path_for(f"{c}_disasters_seed{args.sim_seed}", args.snapshots[0]).exists()]
    for city in cities:
        sid = f"{city}_disasters_seed{args.sim_seed}"
        for S in args.snapshots:
            base_path = output_path_for(sid, S)
            if not base_path.exists():
                raise SystemExit(f"{base_path} not found; run extract_ground_truth.py for {city} T{S}")
            base = checked(base_path)
            for action in args.actions:
                path = output_path_for(sid, S, [action])
                if not path.exists():
                    missing.append(f"{city} T{S} {action}")
                    continue
                do = checked(path)
                used.append(path.name)
                commits |= {line["fbsim_commit"] for line in base + do}
                cells += candidate_cells(base, do, action, horizons)
    if not used:
        raise SystemExit("no do-files found for those cities, snapshots and actions")

    print(f"{len(used)} do-files ({len(missing)} absent: not extracted or infeasible), {len(cells):,} admissible cells")
    weak = weak_pairs(cells)
    cells = [c for c in cells if (c["block"], c["action"], FAMILY[c["qid"]]) not in weak]
    print(f"  weak pairs dropped: {sorted((a, f) for _, a, f in weak) or 'none'}")
    print("  by block:", dict(collections.Counter(c["block"] for c in cells)))
    chosen, notes = draw(cells, args.n, quota, args.seed)
    for note in notes:
        print("  shortfall:", note)
    commit = ", ".join(sorted(commits))
    records = [finish(c, i, args.seed, commit) for i, c in enumerate(chosen)]

    out = Path(args.out) if args.out else g.DATA_DIR / "intcond" / f"intcond_{args.n}_seed{args.seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)

    print(f"\n{len(records)} cells -> {out}")
    tab = collections.Counter((r["snapshot_turn"], r["horizon"], r["block"]) for r in records)
    print("  S    H    " + "  ".join(f"{b:>14}" for b in BLOCKS))
    for S, h in sorted({(s, h) for s, h, _ in tab}):
        print(f"  {S:<4} {h:<4} " + "  ".join(f"{tab[S, h, b]:>14}" for b in BLOCKS))
    d = [abs(r["delta"]) for r in records]
    q = sorted(d)
    print(f"  |delta| q25/50/75/90: {q[len(q) // 4]:.3f}/{statistics.median(d):.3f}/"
          f"{q[3 * len(q) // 4]:.3f}/{q[9 * len(q) // 10]:.3f}; >0.1 {sum(x > .1 for x in d) / len(d):.0%},"
          f" >0.2 {sum(x > .2 for x in d) / len(d):.0%}")
    for b in BLOCKS:
        db = [abs(r["delta"]) for r in records if r["block"] == b]
        if db:
            print(f"    {b:15s} n={len(db):3d} median |delta| {statistics.median(db):.3f}")
    print("  actions:", dict(sorted(collections.Counter(r["action"] for r in records).items())))
    print("  cities:", dict(sorted(collections.Counter(r["city"] for r in records).items())))


if __name__ == "__main__":
    main()
