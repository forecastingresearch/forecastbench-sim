"""Unit tests for the batched continuous eval.

Covers the batch prompt builder, the batch response parser, and the cache
layout helpers. Nothing here calls a model or writes to disk; the only I/O is
synthetic prompt text; corpus tests use invented cached log rows.
"""

from pathlib import Path
from typing import ClassVar

import pytest

import micropolis_world.module_globals as g
from micropolis_world.config import (
    QUESTION_TAGGING_NUMERIC,
    QUESTION_TAGGING_SEMANTIC,
    QUESTIONS_SORT_METRIC,
)
from micropolis_world.continuous_eval import (
    batch_id_for,
    group_into_batches,
    response_path,
)
from micropolis_world.scenarios import (
    DEFAULT_EPILOGUE_PATH,
    DEFAULT_SEMANTIC_EPILOGUE_PATH,
    PERCENTILE_KEYS,
    build_batch_prompt_continuous,
    build_corpus,
    get_base_scenarios,
    parse_batch_percentiles,
    parse_batch_percentiles_semantic,
)
from micropolis_world.templates import ALL_TEMPLATES, asked_templates

FIXTURES_DIR = Path(__file__).parent / "fixtures"

REPORT = "city report: population 1234, funds 5000"

QUESTIONS = [
    {
        "question_id": f"bruce_s42_T240_H48_metric{i}",
        "question_text": f"What will metric {i} be at turn 288?",
        "scenario_id": "bruce_s42",
        "snapshot_turn": 240,
        "semantic_tag": f"metric {i}@288",
        "resolution_turn": 288,
    }
    for i in range(1, 4)
]
TAGS = [q["semantic_tag"] for q in QUESTIONS]
LABELS = [f"test-model {q['question_id']}" for q in QUESTIONS]

SET1 = {"p10": 5.0, "p25": 10.0, "p50": 15.0, "p75": 20.0, "p90": 25.0}
SET2 = {"p10": 100.0, "p25": 200.0, "p50": 300.0, "p75": 400.0, "p90": 500.0}
SET3 = {"p10": 1.0, "p25": 2.0, "p50": 3.0, "p75": 4.0, "p90": 5.0}


def as_line(s: dict) -> str:
    return ", ".join(f"{k}={v:g}" for k, v in s.items())


def delimited(*lines: str) -> str:
    """A response in the requested format, with reasoning above the block."""
    return (
        "Considering the report, p50 is probably around 7 for most of these.\n"
        "<<<PERCENTILES>>>\n" + "\n".join(lines) + "\n<<<END>>>"
    )


class TestParseBatchPercentiles:
    def test_q_numbered_lines(self):
        response = delimited(
            f"Q1: {as_line(SET1)}", f"Q2: {as_line(SET2)}", f"Q3: {as_line(SET3)}"
        )
        assert parse_batch_percentiles(response, LABELS) == [SET1, SET2, SET3]

    def test_dot_and_paren_numbered_lines(self):
        response = delimited(
            f"1. {as_line(SET1)}", f"2) {as_line(SET2)}", f"3: {as_line(SET3)}"
        )
        assert parse_batch_percentiles(response, LABELS) == [SET1, SET2, SET3]

    def test_plain_lines_fall_back_to_positional(self):
        response = delimited(as_line(SET1), as_line(SET2), as_line(SET3))
        assert parse_batch_percentiles(response, LABELS) == [SET1, SET2, SET3]

    def test_numbered_lines_ignore_order_and_skips(self):
        # Q2 is skipped and the others arrive out of order: the stated numbers,
        # not the positions, decide where each answer lands.
        response = delimited(f"Q3: {as_line(SET3)}", f"Q1: {as_line(SET1)}")
        assert parse_batch_percentiles(response, LABELS, quiet=True) == [
            SET1,
            None,
            SET3,
        ]

    def test_numbered_bare_numbers(self):
        response = delimited("Q1: 5, 10, 15, 20, 25", f"Q2: {as_line(SET2)}")
        assert parse_batch_percentiles(response, LABELS, quiet=True) == [
            SET1,
            SET2,
            None,
        ]

    def test_non_monotonic_set_dropped_alone(self):
        response = delimited(
            f"Q1: {as_line(SET1)}",
            "Q2: p10=500, p25=400, p50=300, p75=200, p90=100",
            f"Q3: {as_line(SET3)}",
        )
        assert parse_batch_percentiles(response, LABELS, quiet=True) == [
            SET1,
            None,
            SET3,
        ]

    def test_empty_response(self):
        assert parse_batch_percentiles(None, LABELS, quiet=True) == [None] * 3
        assert parse_batch_percentiles("", LABELS, quiet=True) == [None] * 3

    def test_no_block_still_parses_numbered_lines(self):
        response = f"Q1: {as_line(SET1)}\nQ2: {as_line(SET2)}\nQ3: {as_line(SET3)}"
        assert parse_batch_percentiles(response, LABELS) == [SET1, SET2, SET3]

    def test_single_question_uses_single_format(self):
        response = delimited(as_line(SET1))
        assert parse_batch_percentiles(response, LABELS[:1]) == [SET1]

    def test_prose_numbers_are_not_answers(self):
        # A numbered list in the reasoning must not be read as answers when the
        # real answers never come.
        response = (
            "1. The population will likely grow.\n"
            "2. Funds are stable near 5000, 6000, 7000, 8000, 9000 range.\n"
        )
        parsed = parse_batch_percentiles(response, LABELS, quiet=True)
        assert parsed == [None, None, None]

    def test_restated_format_block_in_reasoning_is_ignored(self):
        # A reasoning model reminding itself of the output format mid-thought
        # emits a whole placeholder block. The real answers come after it, so
        # the last block wins; taking the first read "p10=X" as the answer and
        # reported every question unanswered.
        response = (
            "Let me think about the population trend.\n"
            "Format:\n"
            "<<<PERCENTILES>>>\n"
            "Q1: p10=X, p25=Y, p50=Z, p75=A, p90=B\n"
            "...\n"
            "<<<END>>>\n"
            "Okay, ready to write.\n"
            "<<<PERCENTILES>>>\n"
            f"Q1: {as_line(SET1)}\n"
            f"Q2: {as_line(SET2)}\n"
            f"Q3: {as_line(SET3)}\n"
            "<<<END>>>"
        )
        assert parse_batch_percentiles(response, LABELS) == [SET1, SET2, SET3]

    def test_lone_end_marker_keeps_everything_above_it(self):
        # No opening tag: the answers are prose above a stray <<<END>>>, so the
        # fallback has to span the whole text, not stop at the first line.
        response = (
            "Reasoning about the report.\n"
            f"Q1: {as_line(SET1)}\n"
            f"Q2: {as_line(SET2)}\n"
            f"Q3: {as_line(SET3)}\n"
            "<<<END>>>"
        )
        assert parse_batch_percentiles(response, LABELS) == [SET1, SET2, SET3]

    def test_synthetic_nine_question_response(self):
        response = "\n".join(f"Q{i}: {as_line(SET1)}" for i in range(1, 10))
        assert parse_batch_percentiles(response, [f"q{i}" for i in range(9)]) == [SET1] * 9


class TestBuildBatchPrompt:
    def test_multi_question_prompt(self):
        prompt = build_batch_prompt_continuous(REPORT, QUESTIONS)
        assert prompt.count(REPORT) == 1
        assert "## Questions" in prompt
        for i, q in enumerate(QUESTIONS, 1):
            assert f"{i}. {q['question_text']}" in prompt
        assert "Return 3 numbered answers" in prompt
        assert "p10, p25, p50, p75, p90" in prompt

    def test_single_question_prompt(self):
        # A one-question batch takes the same numbered form as any other, so
        # there is a single prompt shape for the parser to read back.
        prompt = build_batch_prompt_continuous(REPORT, QUESTIONS[:1])
        assert "## Questions" in prompt
        assert f"1. {QUESTIONS[0]['question_text']}" in prompt
        assert "Return 1 numbered answers" in prompt

    def test_epilogue_path_overrides_the_default(self, tmp_path):
        epilogue = tmp_path / "epilogue-test.txt"
        epilogue.write_text("Answer all {n} of them. {n} lines, please.")
        prompt = build_batch_prompt_continuous(
            REPORT, QUESTIONS, epilogue_path=epilogue
        )
        assert prompt.endswith("Answer all 3 of them. 3 lines, please.")
        assert "<<<PERCENTILES>>>" not in prompt  # the default epilogue is gone

    def test_epilogue_without_placeholder_is_used_verbatim(self, tmp_path):
        epilogue = tmp_path / "epilogue-no-n.txt"
        epilogue.write_text('Reply as {"p50": 1}.')
        prompt = build_batch_prompt_continuous(
            REPORT, QUESTIONS, epilogue_path=epilogue
        )
        assert prompt.endswith('Reply as {"p50": 1}.')

    def test_default_epilogue_file_matches_the_prompt(self):
        # The shipped default is what a config gets when it names no
        # epilogue_path, so the prompt must end with it.
        assert build_batch_prompt_continuous(REPORT, QUESTIONS).endswith(
            DEFAULT_EPILOGUE_PATH.read_text(encoding="utf-8").replace("{n}", "3")
        )


@pytest.mark.usefixtures("synthetic_corpus")
class TestQuestionsSort:
    """The two orders build_corpus can number a batch's questions in."""

    HORIZONS: ClassVar = [12, 24]
    # build_corpus censors city funds by default, so the expected order runs
    # over the templates it actually asks — not all of ALL_TEMPLATES.
    TEMPLATES: ClassVar = asked_templates(censor_city_funds=True)

    def _corpus(self, **kw):
        scenarios = get_base_scenarios(seed=42, cities=["bruce"], disasters=[False])
        return build_corpus(
            scenarios,
            [240],
            self.HORIZONS,
            history_freq=12,
            label="sorttest",
            **kw,
        )

    def test_turn_order_is_the_default(self):
        pairs = [(q["horizon"], q["metric"]) for q in self._corpus()]
        assert pairs == [
            (h, t.signal_name) for h in self.HORIZONS for t in self.TEMPLATES
        ]

    def test_metric_order_groups_horizons_per_metric(self):
        corpus = self._corpus(questions_sort=QUESTIONS_SORT_METRIC)
        pairs = [(q["horizon"], q["metric"]) for q in corpus]
        assert pairs == [
            (h, t.signal_name) for t in self.TEMPLATES for h in self.HORIZONS
        ]

    def test_both_orders_ask_the_same_questions(self):
        # Only the numbering changes, so a reordered run must resolve to the
        # same values — otherwise the two orders would not be comparable.
        by_turn = self._corpus()
        by_metric = self._corpus(questions_sort=QUESTIONS_SORT_METRIC)
        assert {q["question_id"]: q["value"] for q in by_turn} == {
            q["question_id"]: q["value"] for q in by_metric
        }

    def test_unknown_sort_is_rejected(self):
        with pytest.raises(ValueError, match="questions_sort"):
            self._corpus(questions_sort="sideways")

    def test_censoring_drops_the_city_funds_questions(self):
        asked = {q["metric"] for q in self._corpus()}
        assert g.FUNDS_METRIC not in asked
        assert asked == {
            t.signal_name for t in ALL_TEMPLATES if t.signal_name != g.FUNDS_METRIC
        }

    def test_not_censoring_asks_about_city_funds(self):
        asked = {q["metric"] for q in self._corpus(censor_city_funds=False)}
        assert asked == {t.signal_name for t in ALL_TEMPLATES}


class TestSemanticTagging:
    """Questions tagged "<metric label>@<turn>" instead of numbered."""

    def test_prompt_lists_tags_instead_of_numbers(self):
        prompt = build_batch_prompt_continuous(
            REPORT, QUESTIONS, question_tagging=QUESTION_TAGGING_SEMANTIC
        )
        for q in QUESTIONS:
            assert f"{q['semantic_tag']} — {q['question_text']}" in prompt
        assert "\n1. " not in prompt  # numbering is gone entirely

    def test_semantic_prompt_uses_the_semantic_epilogue(self):
        # The default epilogue asks for "Q<n>:" lines, which a semantic prompt
        # must not show, so semantic runs pick up epilogue2.txt on their own.
        prompt = build_batch_prompt_continuous(
            REPORT, QUESTIONS, question_tagging=QUESTION_TAGGING_SEMANTIC
        )
        assert prompt.endswith(
            DEFAULT_SEMANTIC_EPILOGUE_PATH.read_text(encoding="utf-8").replace(
                "{n}", "3"
            )
        )

    def test_explicit_epilogue_still_wins(self, tmp_path):
        epilogue = tmp_path / "custom.txt"
        epilogue.write_text("Answer {n} of them.")
        prompt = build_batch_prompt_continuous(
            REPORT,
            QUESTIONS,
            epilogue_path=epilogue,
            question_tagging=QUESTION_TAGGING_SEMANTIC,
        )
        assert prompt.endswith("Answer 3 of them.")

    def test_numeric_tagging_is_the_default(self):
        assert build_batch_prompt_continuous(
            REPORT, QUESTIONS
        ) == build_batch_prompt_continuous(
            REPORT, QUESTIONS, question_tagging=QUESTION_TAGGING_NUMERIC
        )

    def test_unknown_tagging_is_rejected(self):
        with pytest.raises(ValueError, match="question_tagging"):
            build_batch_prompt_continuous(
                REPORT, QUESTIONS, question_tagging="sideways"
            )

    def test_parses_tagged_lines(self):
        response = delimited(
            f"{TAGS[0]}: {as_line(SET1)}",
            f"{TAGS[1]}: {as_line(SET2)}",
            f"{TAGS[2]}: {as_line(SET3)}",
        )
        assert parse_batch_percentiles_semantic(response, LABELS, TAGS) == [
            SET1,
            SET2,
            SET3,
        ]

    def test_answers_may_come_in_any_order(self):
        # The reason semantic tagging exists: order carries no meaning, so a
        # model answering backwards still has each answer land on its question.
        response = delimited(
            f"{TAGS[2]}: {as_line(SET3)}",
            f"{TAGS[0]}: {as_line(SET1)}",
            f"{TAGS[1]}: {as_line(SET2)}",
        )
        assert parse_batch_percentiles_semantic(response, LABELS, TAGS) == [
            SET1,
            SET2,
            SET3,
        ]

    def test_skipped_question_does_not_shift_the_others(self):
        response = delimited(
            f"{TAGS[0]}: {as_line(SET1)}", f"{TAGS[2]}: {as_line(SET3)}"
        )
        assert parse_batch_percentiles_semantic(response, LABELS, TAGS, quiet=True) == [
            SET1,
            None,
            SET3,
        ]

    def test_tag_matching_ignores_case_and_spacing(self):
        response = delimited(f"  Metric 1 @ 288 : {as_line(SET1)}")
        assert parse_batch_percentiles_semantic(response, LABELS, TAGS, quiet=True) == [
            SET1,
            None,
            None,
        ]

    def test_unknown_tag_is_ignored_not_guessed(self):
        # A tag naming no question — including the right metric at the wrong
        # turn — must not be assigned to a question by position.
        response = delimited(
            f"metric 9@288: {as_line(SET1)}", f"metric 1@240: {as_line(SET2)}"
        )
        assert parse_batch_percentiles_semantic(response, LABELS, TAGS, quiet=True) == [
            None,
            None,
            None,
        ]

    def test_no_positional_fallback(self):
        # Unlike the numeric parser, an untagged full set answers nothing:
        # there is no trustworthy way to tell which question it meant.
        response = delimited(as_line(SET1), as_line(SET2), as_line(SET3))
        assert parse_batch_percentiles_semantic(response, LABELS, TAGS, quiet=True) == [
            None,
            None,
            None,
        ]

    def test_restated_answer_wins(self):
        response = (
            f"{TAGS[0]}: {as_line(SET2)}\n"
            "<<<PERCENTILES>>>\n"
            f"{TAGS[0]}: {as_line(SET1)}\n"
            "<<<END>>>"
        )
        assert (
            parse_batch_percentiles_semantic(response, LABELS, TAGS, quiet=True)[0]
            == SET1
        )

    def test_non_monotonic_is_discarded(self):
        response = delimited(f"{TAGS[0]}: p10=50, p25=40, p50=30, p75=20, p90=10")
        assert parse_batch_percentiles_semantic(response, LABELS, TAGS, quiet=True) == [
            None,
            None,
            None,
        ]

    def test_empty_response(self):
        assert (
            parse_batch_percentiles_semantic(None, LABELS, TAGS, quiet=True)
            == [None] * 3
        )
        assert (
            parse_batch_percentiles_semantic("", LABELS, TAGS, quiet=True) == [None] * 3
        )


class TestQuestionsPerPrompt:
    """Splitting a scenario's questions across several prompts."""

    def _corpus(self, n, snapshots=(240,)):
        return [
            {
                "scenario_id": "bruce_s42",
                "snapshot_turn": t,
                "question_id": f"q{t}_{i}",
            }
            for t in snapshots
            for i in range(n)
        ]

    def test_default_keeps_one_prompt_per_batch(self):
        batches = group_into_batches(self._corpus(20))
        assert {k: len(v) for k, v in batches.items()} == {"bruce_s42_T240": 20}

    def test_cap_at_or_above_batch_size_does_not_split(self):
        # The plain batch id must survive, or every existing cache entry would
        # be orphaned the moment a cap was set generously.
        for cap in (20, 21, 999):
            assert list(group_into_batches(self._corpus(20), cap)) == ["bruce_s42_T240"]

    def test_split_ids_are_suffixed(self):
        batches = group_into_batches(self._corpus(20), 12)
        assert list(batches) == ["bruce_s42_T240_c1of2", "bruce_s42_T240_c2of2"]

    def test_chunks_are_evened_out(self):
        # 20 capped at 12 is two prompts, so they split 10/10 rather than
        # filling one to 12 and leaving 8 in the other.
        batches = group_into_batches(self._corpus(20), 12)
        assert [len(v) for v in batches.values()] == [10, 10]

    def test_no_chunk_exceeds_the_cap(self):
        for n in (13, 20, 25, 36, 37, 72):
            sizes = [len(v) for v in group_into_batches(self._corpus(n), 12).values()]
            assert max(sizes) <= 12
            assert sum(sizes) == n
            assert max(sizes) - min(sizes) <= 1  # evened out

    def test_questions_are_neither_lost_nor_reordered(self):
        corpus = self._corpus(20)
        batches = group_into_batches(corpus, 7)
        flat = [q for v in batches.values() for q in v]
        assert flat == corpus

    def test_each_snapshot_splits_independently(self):
        batches = group_into_batches(self._corpus(20, snapshots=(240, 480)), 12)
        assert list(batches) == [
            "bruce_s42_T240_c1of2",
            "bruce_s42_T240_c2of2",
            "bruce_s42_T480_c1of2",
            "bruce_s42_T480_c2of2",
        ]

    def test_cap_of_one_gives_a_prompt_per_question(self):
        batches = group_into_batches(self._corpus(3), 1)
        assert [len(v) for v in batches.values()] == [1, 1, 1]

    def test_each_chunk_is_a_self_contained_prompt(self):
        # A chunk repeats the report and renumbers from 1, and its epilogue
        # states its own count — the model never sees a question number it was
        # not asked about, nor a count that disagrees with the list.
        questions = [
            dict(q, question_text=f"What about {i}?", context=REPORT)
            for i, q in enumerate(self._corpus(6))
        ]
        chunks = list(group_into_batches(questions, 4).values())
        assert [len(c) for c in chunks] == [3, 3]
        for chunk in chunks:
            prompt = build_batch_prompt_continuous(chunk[0]["context"], chunk)
            assert prompt.count(REPORT) == 1
            assert "1. What about" in prompt
            assert "Return 3 numbered answers" in prompt

    def test_zero_is_rejected(self):
        with pytest.raises(ValueError, match="questions_per_prompt"):
            group_into_batches(self._corpus(20), 0)


class TestBatchCacheHelpers:
    def test_batch_id_groups_by_scenario_and_snapshot(self):
        assert batch_id_for(QUESTIONS[0]) == "bruce_s42_T240"
        other_snapshot = dict(QUESTIONS[0], snapshot_turn=480)
        other_scenario = dict(QUESTIONS[0], scenario_id="kobe_s42")
        assert batch_id_for(other_snapshot) == "bruce_s42_T480"
        assert batch_id_for(other_scenario) == "kobe_s42_T240"

    def test_response_path_sanitizes_model_id(self):
        path = response_path("bruce_s42_T240", "openai/gpt-4o-2024-05-13", "abcd1234")
        assert path.name == "response-openai_gpt-4o-2024-05-13-abcd1234.txt"
        assert path.parent.name == "bruce_s42_T240"

    def test_response_path_escapes_a_slug_suffix(self):
        path = response_path("bruce_s42_T240", "openai/o3:lowef", "abcd1234")
        assert path.name == "response-openai_o3+lowef-abcd1234.txt"
