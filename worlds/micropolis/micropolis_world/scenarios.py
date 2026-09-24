"""Scenario sampler + corpus builder."""

import json
import re
from functools import cache
from itertools import pairwise, product
from pathlib import Path

from fbsim_core.questions.resolver import QuestionResolver
from fbsim_core.questions.schema import QuestionInstance

from . import module_globals as g
from .city_sim import CitySimulation, to_world
from .config import (
    DATAFILES_DIR,
    QUESTION_TAGGING_NUMERIC,
    QUESTION_TAGGING_SEMANTIC,
    QUESTION_TAGGINGS,
    QUESTIONS_SORT_TURN,
    QUESTIONS_SORTS,
)
from .report import gen_world_report
from .templates import REGISTRY, asked_templates

# The quantiles elicited for every continuous question, matching FreeCiv.
PERCENTILE_KEYS = ["p10", "p25", "p50", "p75", "p90"]

# to_world() keys entities by their position in the dict it is passed, so the
# single city in each scenario is always entity 0.
CITY_ENTITY_ID = 0

# The preamble a run uses when its config names no "preamble_path". Held as a
# file rather than a string literal so a prompt variant is a new file plus one
# config key, with no code change; the text must contain "{sources}", which
# prompt_preamble fills in.
DEFAULT_PREAMBLE_PATH = DATAFILES_DIR / "preamble1.txt"

# The epilogue a run uses when its config names no "epilogue_path". Held as a
# file for the same reason as the preamble; the text may contain "{n}", which
# read_epilogue fills in with the number of questions in the batch.
DEFAULT_EPILOGUE_PATH = DATAFILES_DIR / "epilogue1.txt"

# The epilogue a semantic-tagging run uses when its config names no
# "epilogue_path": the default one shows "Q1:" answer lines, which are not what
# a semantic prompt asks for.
DEFAULT_SEMANTIC_EPILOGUE_PATH = DATAFILES_DIR / "epilogue2.txt"

# Separates a semantic question's tag from its text. An em dash rather than a
# hyphen so it cannot be confused with a minus sign in the question itself.
SEMANTIC_TAG_SEPARATOR = " — "

# Defaults for the binary yes/no eval's prompt, mirroring the continuous pair.
DEFAULT_BINARY_PREAMBLE_PATH = DATAFILES_DIR / "preamble-binary.txt"
DEFAULT_BINARY_EPILOGUE_PATH = DATAFILES_DIR / "epilogue-binary.txt"

# The answer-block marker a binary response is asked for; accepts the singular
# form too, matching FreeCiv's parser tolerance.
_PROBABILITIES_MARKER_RE = r"PROBABILIT(?:Y|IES)"


def semantic_tag(metric: str, resolution_turn: int) -> str:
    """The "<metric label>@<turn>" tag naming one question in semantic mode.

    Built here rather than at each call site so the tag written into the prompt
    and the tag the parser matches against can never drift apart.
    """
    return f"{g.METRIC_LABELS[metric]}@{resolution_turn}"


@cache
def read_preamble(path: Path | str | None = None) -> str:
    """The raw preamble template at `path`, or the default one when None.

    Cached because the corpus and eval scripts ask for the same preamble once
    per batch, and because a config's key is read afresh at each call site.
    """
    return Path(path or DEFAULT_PREAMBLE_PATH).read_text(encoding="utf-8")


@cache
def _read_epilogue_template(path: Path | str | None = None) -> str:
    """The raw epilogue template at `path`, or the default one when None."""
    return Path(path or DEFAULT_EPILOGUE_PATH).read_text(encoding="utf-8")


def read_epilogue(n: int, path: Path | str | None = None) -> str:
    """The epilogue at `path` with "{n}" replaced by the question count.

    A plain str.replace rather than str.format so an epilogue that never
    mentions the count — or one that contains braces of its own, such as a
    JSON answer template — is returned untouched instead of raising.
    """
    return _read_epilogue_template(path).replace("{n}", str(n))


def get_base_scenarios(
    seed: int, cities: list[str], disasters: list[bool]
) -> list[CitySimulation]:
    scenarios = []
    for city, has_disasters in product(cities, disasters):
        sim = CitySimulation(city_name=city, seed=seed, disasters=has_disasters)
        scenarios.append(sim)
    return scenarios


def build_corpus(
    scenarios: list[CitySimulation],
    snapshot_turns: list[int],
    horizons: list[int],
    history_freq: int,
    label: str,
    snapshot_only_report: bool = False,
    history_length: int = -1,
    report_effectiveness: bool = False,
    censor_city_funds: bool = True,
    questions_sort: str = QUESTIONS_SORT_TURN,
) -> list[dict]:
    """One question per (scenario, snapshot turn, horizon, metric) combination.

    `censor_city_funds` (the default) both hides the city's money from the
    world report and drops the city funds question from the corpus, so the
    metric is neither reported nor asked about. False asks about all of
    templates.Q_METRICS and reports the balance, which is what the prompt
    variants gathered before the flag existed were run with.

    `questions_sort` picks the order the questions come out in, and so the
    order they are numbered in each batch prompt: QUESTIONS_SORT_TURN asks
    every metric for one horizon before moving to the next horizon, while
    QUESTIONS_SORT_METRIC asks every horizon for one metric first. Both ask
    exactly the same questions; only the numbering differs, and since that
    order is part of the prompt text, switching hashes to a new cache entry
    rather than mixing variants.
    """
    if questions_sort not in QUESTIONS_SORTS:
        raise ValueError(
            f"questions_sort must be one of {QUESTIONS_SORTS}, got {questions_sort!r}"
        )
    resolver = QuestionResolver(REGISTRY)
    templates = asked_templates(censor_city_funds)
    corpus = []
    nscenarios = len(scenarios)
    # +1 because the furthest question resolves *at* max(snapshot_turns) +
    # max(horizons), and a run of N turns only covers indices 0..N-1.
    nturns = max(snapshot_turns) + max(horizons) + 1
    for i, sim in enumerate(scenarios):
        sim.run_if_needed_and_load(nturns=nturns, quiet=True)

        world = to_world({"city1": sim})

        scenario_id = sim.get_id_str()
        for SNAPSHOT_TURN in snapshot_turns:
            report_text = gen_world_report(
                sim,
                turn=SNAPSHOT_TURN,
                history_freq=history_freq,
                label=label,
                snapshot_only=snapshot_only_report,
                history_length=history_length,
                report_effectiveness=report_effectiveness,
                censor_city_funds=censor_city_funds,
            )
            # Both orders ask the same questions; only the order they are
            # numbered in the prompt differs, so the pairs are generated once
            # and re-ordered rather than duplicating the loop body.
            pairs = (
                [(H, t) for H in horizons for t in templates]
                if questions_sort == QUESTIONS_SORT_TURN
                else [(H, t) for t in templates for H in horizons]
            )
            for H, template in pairs:
                T = SNAPSHOT_TURN + H
                q_text = template.question_template.format(resolution_turn=T)
                question_id = (
                    f"{scenario_id}_T{SNAPSHOT_TURN}_H{H}_{template.template_id}"
                )
                q = QuestionInstance(
                    question_id=question_id,
                    template_id=template.template_id,
                    resolution_turn=T,
                    horizon=H,
                    parameters={
                        "player_id": CITY_ENTITY_ID,
                    },
                    question_text=q_text,
                )

                res = resolver.resolve(q, world, SNAPSHOT_TURN)
                # A continuous question resolves to a number in value_at_resolution;
                # .answer is only a bool saying whether any data was found. A missing
                # entity id or an out-of-range turn silently yields None here, which
                # would otherwise be indistinguishable from a real result.
                if res.value_at_resolution is None:
                    raise ValueError(
                        f"{q.question_id}: no value for {template.signal_name} at turn "
                        f"{T} (entity {CITY_ENTITY_ID}, run has {sim.nturns} turns)"
                    )
                entry = {
                    "question_id": q.question_id,
                    "metric": template.signal_name,
                    # Carried on the entry so the prompt builder and the
                    # response parser derive a question's semantic tag from
                    # the same place; unused when tagging is numeric.
                    "semantic_tag": semantic_tag(template.signal_name, T),
                    "resolution_turn": T,
                    "snapshot_turn": SNAPSHOT_TURN,
                    "horizon": H,  # TODO: this should be one of "H0", "H1", ...
                    "scenario_id": scenario_id,
                    "question_text": q_text,
                    "value": res.value_at_resolution,
                    "context": report_text,
                    "scenario": sim.describe(),
                }

                corpus.append(entry)
    ntotal_questions = len(corpus)
    questions_per_scenario = ntotal_questions / nscenarios
    print(
        f"\nbuild_corpus: corpus with {ntotal_questions} questions for {nscenarios} scenarios generated ({questions_per_scenario} questions per scenario)."
    )
    return corpus


def build_batch_prompt_continuous(
    context: str,
    questions: list[dict],
    preamble_path: Path | str | None = None,
    epilogue_path: Path | str | None = None,
    question_tagging: str = QUESTION_TAGGING_NUMERIC,
) -> str:
    """Ask for one p10/p25/p50/p75/p90 quantile forecast per question.

    Every question in `questions` shares the game report in `context`, so the
    report — which dominates the token cost — is included once and the
    questions are numbered. The instruction wording and the delimited answer
    block match FreeCiv's build_continuous_batch_prompt, so responses from the
    two worlds are parsed the same way and scored on the same CRPS.
    parse_batch_percentiles reads the answers back.

    `question_tagging` picks how each question is labeled, and so how its
    answer is matched back to it: QUESTION_TAGGING_NUMERIC numbers them "1.",
    "2.", ...; QUESTION_TAGGING_SEMANTIC prefixes each with its
    "<metric label>@<turn>" tag instead, which lets a model answer out of order
    without its answers sliding onto the wrong questions.

    `preamble_path` and `epilogue_path` name the templates wrapping the report
    and questions, defaulting to DEFAULT_PREAMBLE_PATH and — since the default
    epilogue asks for the numeric answer format — to whichever epilogue matches
    `question_tagging`. All three are part of the prompt, so changing any of
    them misses the response cache rather than mixing variants.
    """
    if question_tagging not in QUESTION_TAGGINGS:
        raise ValueError(
            f"question_tagging must be one of {QUESTION_TAGGINGS}, "
            f"got {question_tagging!r}"
        )
    n = len(questions)
    if question_tagging == QUESTION_TAGGING_SEMANTIC:
        listed_questions = "\n".join(
            f"{q['semantic_tag']}{SEMANTIC_TAG_SEPARATOR}{q['question_text']}"
            for q in questions
        )
        if epilogue_path is None:
            epilogue_path = DEFAULT_SEMANTIC_EPILOGUE_PATH
    else:
        listed_questions = "\n".join(
            f"{i}. {q['question_text']}" for i, q in enumerate(questions, 1)
        )
    return f"""{read_preamble(preamble_path)}

## Game report
{context}

## Questions
{listed_questions}

{read_epilogue(n, epilogue_path)}"""


def build_batch_prompt_binary(
    context: str,
    questions: list[dict],
    preamble_path: Path | str | None = None,
    epilogue_path: Path | str | None = None,
) -> str:
    """Ask for one P(Yes) per binary question, sharing one game report.

    The binary counterpart of build_batch_prompt_continuous: same skeleton,
    numeric tagging only, and the binary preamble/epilogue defaults. Answers
    come back as a <<<PROBABILITIES>>> block of "Q1: 0.65" lines, read by
    parse_batch_probabilities.
    """
    listed_questions = "\n".join(
        f"{i}. {q['question_text']}" for i, q in enumerate(questions, 1)
    )
    return f"""{read_preamble(preamble_path or DEFAULT_BINARY_PREAMBLE_PATH)}

## Game report
{context}

## Questions
{listed_questions}

{read_epilogue(len(questions), epilogue_path or DEFAULT_BINARY_EPILOGUE_PATH)}"""
















# A line answering one question of a batch, e.g. "Q3: p10=...", "3. p10=...",
# or "Question 3) ...". Anchored to the line start: a question number mentioned
# mid-sentence is prose, not an answer.
_QUESTION_NUMBER_RE = re.compile(
    r"^\s*(?:question\s*|q)?(\d+)\s*[.:)]\s*", re.IGNORECASE
)




# A semantic answer line: everything up to the first colon is the tag, the rest
# holds the percentiles. Anchored to the line start, like _QUESTION_NUMBER_RE,
# so a tag mentioned mid-sentence in the reasoning is prose, not an answer. The
# tag is matched against the prompt's tags rather than parsed, so a metric label
# containing a space, an "@", or a digit needs no special handling here.
_SEMANTIC_TAG_RE = re.compile(r"^\s*([^:]+?)\s*:\s*")






# A probability at the start of a string: "0.65", ".65", "1", "65%", "65 %".
# Anchored so a number buried in trailing prose ("fewer than 1 in 10") is not
# read as the answer, while "0.65 (earthquake unlikely)" still is.
_PROB_TOKEN_RE = re.compile(r"^\s*(\d+(?:\.\d+)?|\.\d+)\s*(%?)")






# A parsed probability and the 1-based line of the response it was read from;
# both None when no usable answer was found.
ProbabilityLine = tuple[float | None, int | None]











# Native parser compatibility is maintained by fbsim-benchmark.
from fbsim_benchmark.parsing.micropolis import (
    _at,
    _extract_answer_block,
    _extract_answer_span,
    _normalize_tag,
    _scan_bare_percentiles,
    _scan_labeled_percentiles,
    _scan_probability,
    _validate_monotonic,
    _validate_probability,
    _with_line,
    parse_batch_percentiles,
    parse_batch_percentiles_semantic,
    parse_batch_probabilities,
    parse_batch_probabilities_with_lines,
    parse_percentiles,
    parse_probability,
    parse_probability_with_line,
)
