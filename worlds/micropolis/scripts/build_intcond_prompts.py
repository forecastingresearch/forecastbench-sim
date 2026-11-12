#!/usr/bin/env -S uv run python3
"""Build the turn-1 / turn-2 prompts for an interventional-conditional draw (draw_intcond.py).

Turn 1 is the natural-conditional turn 1; turn 2 adds intervention X. The world
report is the binary eval's cached prompt for each city and snapshot. No provider calls.

Usage:
    scripts/build_intcond_prompts.py                     # intcond_400_seed42.jsonl
    scripts/build_intcond_prompts.py --show 2            # also print two example conversations
"""

import argparse
import collections
import json
from pathlib import Path

from micropolis_world import module_globals as g
from micropolis_world.intcond_prompts import build_jobs
from micropolis_world.natcond_prompts import (
    PREAMBLE_PATH,
    binary_prompt_path,
    conversation,
    report_from_prompt,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draw", type=Path, default=None, help="default: data/micropolis/intcond/intcond_400_seed42.jsonl")
    ap.add_argument("--cache", type=Path, default=None, help="binary-eval prompt cache (default: data/micropolis/binary/cache)")
    ap.add_argument("--out", type=Path, default=None, help="default: <draw stem>_prompts.jsonl beside the draw")
    ap.add_argument("--show", type=int, default=0, help="print this many example conversations, one per block first")
    args = ap.parse_args()
    draw = args.draw or g.DATA_DIR / "intcond" / "intcond_400_seed42.jsonl"
    cache = args.cache or g.DATA_DIR / "binary" / "cache"
    out = args.out or draw.with_name(draw.stem + "_prompts.jsonl")

    with open(draw) as f:
        cells = [json.loads(line) for line in f]
    reports = {key: report_from_prompt(binary_prompt_path(cache, *key).read_text())
               for key in sorted({(c["scenario_id"], c["snapshot_turn"]) for c in cells})}
    jobs = build_jobs(cells, reports, PREAMBLE_PATH.read_text())

    with open(out, "w") as f:
        for arm in ("t1", "t2"):
            f.writelines(json.dumps(j) + "\n" for j in jobs[arm])
    print(f"{len(cells)} cells, {len(reports)} reports -> {out}")
    print("  jobs:", {arm: len(v) for arm, v in jobs.items()}, "per model")

    t1 = {j["question_id"]: j for j in jobs["t1"]}
    by_cell = {c["cell_id"]: c for c in cells}
    by_block = collections.defaultdict(list)
    for j in jobs["t2"]:
        by_block[j["block"]].append(j)
    for j in [lst[0] for lst in by_block.values()][: args.show]:
        c = by_cell[j["cell_id"]]
        print("\n" + "=" * 30, f"cell {j['cell_id']} ({j['block']}, {c['action']}): "
              f"P(A) {c['p_a']:.3f}, P(A|do X) {c['p_a_do']:.3f}", "=" * 30)
        for m in conversation(t1[j["question_id"]], "<the model's turn-1 reply>", j):
            print(f"\n--- {m['role']} ---\n{m['content']}")


if __name__ == "__main__":
    main()
