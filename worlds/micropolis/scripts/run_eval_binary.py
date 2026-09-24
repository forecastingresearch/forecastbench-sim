#!/usr/bin/env -S uv run python3
"""Binary yes/no evaluation: gather model probability forecasts.

The binary counterpart of run_eval_continuous.py: builds the 27-question
corpus from binary_questions.py for every (city, disasters) combination in the
config, prompts each configured model for one P(Yes) per question, and writes
the questions and forecasts to data/micropolis/binary/{label}/data.json.

Questions sharing a game report — same scenario and snapshot turn — are asked
together in one numbered prompt. Prompts and raw responses are cached under
data/micropolis/binary/cache/{batch_id}/ with the same content-addressed
naming as the continuous eval, so a config change misses the cache rather
than mixing variants, and data.json is regenerated from cached responses on
every run — which --cache-only makes the whole run, sending no prompts and
leaving uncached questions out of the dataset. The gather loop below mirrors
run_eval_continuous.gather_responses (minus its semantic-tagging branches); a
third eval variant should extract the shared machinery rather than copy it
again.

Usage:
    scripts/run_eval_binary.py                      # configs/binary.json5
    scripts/run_eval_binary.py my_config.json5 --seed 7
    scripts/run_eval_binary.py --dry-run            # corpus + Yes counts only
    scripts/run_eval_binary.py --cache-only         # no prompts; cached only
    scripts/run_eval_binary.py --cities kyoto bruce
"""

import argparse
from collections import Counter
from pathlib import Path

import micropolis_world.module_globals as g
from micropolis_world.binary_eval import (
    PATHS,
    BinaryResponse,
    BinaryResponses,
    data_path,
    save_dataset_binary,
)
from micropolis_world.binary_questions import QUESTION_IDS, build_corpus_binary
from micropolis_world.config import (
    CONFIG_DIR,
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.continuous_eval import ResponseId
from micropolis_world.gather import gather_raw_responses
from micropolis_world.scenarios import (
    build_batch_prompt_binary,
    get_base_scenarios,
    parse_batch_probabilities_with_lines,
)

DEFAULT_BINARY_CONFIG_PATH = CONFIG_DIR / "example.json5"


def gather_responses_binary(
    corpus: list[dict],
    model_names: list[str],
    concurrency: int | None = None,
    preamble_path: Path | None = None,
    epilogue_path: Path | None = None,
    questions_per_prompt: int = -1,
    cache_only: bool = False,
) -> BinaryResponses:
    """Prompt each model on each batch, then read probabilities out of the replies.

    The gathering itself — batching, the prompt hash cache, the concurrent
    calls, the cost accounting — is gather.gather_raw_responses, shared with
    the continuous eval. What is binary-specific is the prompt this asks for
    and the P(Yes) read back out of it.
    """
    batches, raws, rpaths = gather_raw_responses(
        corpus,
        model_names,
        paths=PATHS,
        build_prompt=lambda context, questions: build_batch_prompt_binary(
            context, questions, preamble_path, epilogue_path
        ),
        concurrency=concurrency,
        questions_per_prompt=questions_per_prompt,
        cache_only=cache_only,
    )

    # Parse after the gather, in the stable model x batch order. A failed call
    # has no raws entry and so gets no response rows at all — "never gathered;
    # re-run", exactly right since re-running retries it.
    responses: BinaryResponses = {}
    for model_name in model_names:
        for bid, questions in batches.items():
            if (bid, model_name) not in raws:
                continue
            raw = raws[(bid, model_name)]
            labels = [f"{model_name} {q['question_id']}" for q in questions]
            # The cache file the text came from, so a warning about an
            # unusable forecast points at the response to go read.
            rpath = rpaths[(bid, model_name)]
            parsed = parse_batch_probabilities_with_lines(raw, labels, source=rpath)
            for q, (probability, line) in zip(questions, parsed):
                responses[ResponseId(model_name, q["question_id"])] = BinaryResponse(
                    actual=q["answer"],
                    probability=probability,
                    response_text=raw,
                    source=PATHS.cache_relative(rpath),
                    line=line,
                )
    return responses


def print_yes_counts(corpus: list[dict]) -> None:
    """Per-question Yes counts, for eyeballing against the doc's P(Yes) ranges."""
    windows = sorted({(c["snapshot_turn"], c["horizon"]) for c in corpus})
    yes = Counter(
        (c["qid"], c["snapshot_turn"], c["horizon"]) for c in corpus if c["answer"]
    )
    nscenarios = len({c["scenario_id"] for c in corpus})
    header = "  ".join(f"T{t}+{h}" for t, h in windows)
    print(f"\nYes counts out of {nscenarios} scenario(s) per window:")
    print(f"  {'':>4} {header}")
    for qid in QUESTION_IDS:
        counts = "  ".join(
            f"{yes[(qid, t, h)]:>{len(f'T{t}+{h}')}}" for t, h in windows
        )
        print(f"  {qid:>4} {counts}")


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default=DEFAULT_BINARY_CONFIG_PATH)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--cache-only",
        action="store_true",
        help="send no prompts; build data.json from already-cached responses",
    )
    args = ap.parse_args()

    cfg = load_config(args)
    seed = cfg.get_seed(args.seed)
    label = cfg.get_label(args.label)
    models = cfg.get_models(args.models)
    out_path = data_path(label)
    censor_city_funds = cfg.get_bool_or("censorCityFunds", True)
    report_census = cfg.get_bool_or("report_census", True)
    preamble_path = cfg.get_preamble_path()
    epilogue_path = cfg.get_epilogue_path()
    questions_per_prompt = cfg.get_questions_per_prompt()

    print("=" * 70)
    print("MICROPOLIS WORLD — binary eval")
    print("=" * 70)
    print(f"config: {cfg.path}")
    print(f"label:  {label}")
    if preamble_path is not None:
        print(f"preamble: {preamble_path}")
    if epilogue_path is not None:
        print(f"epilogue: {epilogue_path}")
    if not report_census:
        print("report: census section omitted")
    if questions_per_prompt > 0:
        print(f"questions per prompt: at most {questions_per_prompt}")

    scenarios = get_base_scenarios(
        seed=seed,
        cities=cfg.get_cities(args.cities),
        disasters=cfg.get_disasters(args.disasters),
    )
    print(f"\nRunning {len(scenarios)} with seed={seed}; building corpus...")
    corpus = build_corpus_binary(
        scenarios,
        cfg.get_int_list("snapshot_turns"),
        cfg.get_int_list("horizons"),
        cfg.get_int("history_freq"),
        label,
        cfg.get_bool_or("snapshot_only_report", False),
        cfg.get_int_or("history_length", -1),
        cfg.get_bool_or("report_effectiveness", False),
        censor_city_funds,
        report_census,
    )

    if args.dry_run:
        print_yes_counts(corpus)
        print("\nDry run. Exiting")
        return

    if args.cache_only:
        print("\nCache-only run: no prompts will be sent")
    else:
        print("\nGathering model responses...")
        g.ensure_api_keys()
    responses = gather_responses_binary(
        corpus,
        models,
        cfg.get_concurrency(),
        preamble_path,
        epilogue_path,
        questions_per_prompt,
        args.cache_only,
    )
    print("Done gathering")

    save_dataset_binary(corpus, responses, models, out_path)
    usable = sum(1 for r in responses.values() if r.probability is not None)
    print(f"\nWrote {len(corpus)} questions x {len(models)} models -> {out_path}")
    print(f"  {usable} of {len(responses)} forecasts usable")
    print("=" * 70)


if __name__ == "__main__":
    main()
