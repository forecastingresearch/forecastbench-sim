#!/usr/bin/env -S uv run python3
"""Print the first prompt run_eval_continuous.py would send for a config.

Builds the corpus exactly as run_eval_continuous.py does — same preamble,
game report, question order, tagging and epilogue — but only for the first
(city, disasters) scenario, which is all the first batch draws on, and prints
that batch's prompt to stdout. Status lines go to stderr, so the output can be
piped or redirected as the bare prompt text.

Useful for eyeballing what a prompt variant actually asks, and for diffing
two configs' prompts against each other.

Usage:
    scripts/generate_prompt.py my_config.json5
    scripts/generate_prompt.py my_config.json5 --seed 7 > prompt.txt
"""

import argparse
import contextlib
import sys

from micropolis_world.config import add_config_args, load_config, main_with_config
from micropolis_world.continuous_eval import group_into_batches
from micropolis_world.scenarios import (
    build_batch_prompt_continuous,
    build_corpus,
    get_base_scenarios,
)


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    args = ap.parse_args()

    cfg = load_config(args)
    seed = cfg.get_seed(args.seed)
    label = cfg.get_label(args.label)

    scenarios = get_base_scenarios(
        seed=seed,
        cities=cfg.get_cities(args.cities),
        disasters=cfg.get_disasters(args.disasters),
    )
    # Only the first scenario: group_into_batches preserves corpus order, so
    # the first batch of the full run comes entirely from it. The snapshot
    # turns and horizons are kept whole, so the simulated run has the same
    # length — and the batch the same questions — as in the full eval.
    with contextlib.redirect_stdout(sys.stderr):
        corpus = build_corpus(
            [scenarios[0]],
            cfg.get_int_list("snapshot_turns"),
            cfg.get_int_list("horizons"),
            cfg.get_int("history_freq"),
            label,
            cfg.get_bool_or("snapshot_only_report", False),
            cfg.get_int_or("history_length", -1),
            cfg.get_bool_or("report_effectiveness", False),
            cfg.get_bool_or("censorCityFunds", True),
            cfg.get_questions_sort(),
        )
        batches = group_into_batches(corpus, cfg.get_questions_per_prompt())
        bid, questions = next(iter(batches.items()))
        print(f"config: {cfg.path}\nbatch:  {bid} ({len(questions)} questions)")
    print(
        build_batch_prompt_continuous(
            questions[0]["context"],
            questions,
            cfg.get_preamble_path(),
            cfg.get_epilogue_path(),
            cfg.get_question_tagging(),
        )
    )


if __name__ == "__main__":
    main()
