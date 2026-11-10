#!/usr/bin/env python3
"""Minimal consumer for run_continuations.js's stdout stream.

Reads JSONL from stdin, parses every line, and at the end prints how many lines
and bytes were read per "kind" (header/begin/stats/event/end) plus totals and
elapsed time. Exists to measure the cost of piping the stream into Python
rather than to detect anything.

    pnpm tsx cli/run_continuations.js -S 1440 -H 480 --branch-seeds 1-100 | python3 cli/count_stream.py
"""

import json
import sys
import time
from collections import Counter


def main() -> None:
    t0 = time.perf_counter()
    lines = Counter()
    nbytes = Counter()
    runs = 0
    for raw in sys.stdin.buffer:
        rec = json.loads(raw)
        kind = rec["kind"]
        lines[kind] += 1
        nbytes[kind] += len(raw)
        if kind == "end":
            runs += 1
    elapsed = time.perf_counter() - t0

    total_lines = sum(lines.values())
    total_bytes = sum(nbytes.values())
    print(f"{'kind':<8}{'lines':>10}{'bytes':>14}")
    for kind in sorted(lines, key=lambda k: -nbytes[k]):
        print(f"{kind:<8}{lines[kind]:>10}{nbytes[kind]:>14}")
    print(f"{'total':<8}{total_lines:>10}{total_bytes:>14}")
    print(f"runs (end lines): {runs}")
    print(f"reader elapsed: {elapsed:.1f} s  ({total_bytes / elapsed / 1e6:.1f} MB/s, {total_lines / elapsed / 1e3:.0f}k lines/s)")


if __name__ == "__main__":
    main()
