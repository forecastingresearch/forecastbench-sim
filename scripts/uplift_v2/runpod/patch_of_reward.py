#!/usr/bin/env python3
"""Patch OpenForecaster's vendored verl binary reward for the dense-vs-hard A/B.

Two changes, applied to verl/trainer/verifier.py and verl/utils/reward_score/binary.py:
1. Generalize the binary Brier to (p - resolution)^2 so a float p_mc target works.
   Identical to upstream behavior when resolution is 0/1.
2. Fix the falsy-zero parse bug (`if probability and ...` treats a parsed 0.0 as
   unparseable and applies the -1 format penalty). This asymmetrically punishes the
   hard arm (whose optimum is extreme probabilities), so it would confound the A/B.

Exact-match replacement: exits 1 loudly if any expected snippet is missing.

Usage: python patch_of_reward.py /path/to/libraries/verl
"""
import sys
from pathlib import Path

VERL = Path(sys.argv[1])

BRIER_BODY_OLD = """    if resolution == 1:
        # If answer is correct: -(1 - p)^2
        return ((1 - probability) ** 2)
    else:
        # If answer is incorrect: -(1 + p^2)
        return  (probability ** 2)
        # return - (probability ** 2)"""
BRIER_BODY_NEW = """    # Generalized: squared error vs a target in [0,1]. Identical to the original
    # when resolution is 0/1; supports dense p_mc targets (civbench A/B patch).
    return (probability - float(resolution)) ** 2"""

FALSY_OLD_VERIFIER = "if probability and probability >= -0.01 and probability <= 1.01:"
FALSY_OLD_BINARY = "if probability and probability >= -0.1 and probability <= 1.1:"
FALSY_NEW_VERIFIER = ("if probability is not None and probability >= -0.01 "
                      "and probability <= 1.01:")
FALSY_NEW_BINARY = ("if probability is not None and probability >= -0.1 "
                    "and probability <= 1.1:")


def patch(path: Path, pairs: list[tuple[str, str]]) -> None:
    text = path.read_text()
    for old, new in pairs:
        n = text.count(old)
        if n == 0:
            sys.exit(f"PATCH FAILED: pattern not found in {path}:\n{old}")
        text = text.replace(old, new)
        print(f"  {path.name}: replaced {n} occurrence(s) of "
              f"{old.strip().splitlines()[0][:60]}...")
    path.write_text(text)


verifier = VERL / "verl/trainer/verifier.py"
binary = VERL / "verl/utils/reward_score/binary.py"
# verifier.py: calculate_brier_score_binary body (appears once inside that function —
# the freeform calculate_brier_score has a different body with -(1 + ...)).
patch(verifier, [(BRIER_BODY_OLD, BRIER_BODY_NEW),
                 (FALSY_OLD_VERIFIER, FALSY_NEW_VERIFIER)])
patch(binary, [(BRIER_BODY_OLD, BRIER_BODY_NEW),
               (FALSY_OLD_BINARY, FALSY_NEW_BINARY)])
print("PATCH OK")
