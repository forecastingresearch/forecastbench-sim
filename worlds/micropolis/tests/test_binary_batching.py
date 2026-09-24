"""Unit tests for the binary batch prompt builder and probability parser.

Nothing here calls a model, writes to disk, or runs the simulator.
"""

from micropolis_world import binary_eval
from micropolis_world.scenarios import (
    _extract_answer_block,
    build_batch_prompt_binary,
    parse_batch_probabilities,
    parse_batch_probabilities_with_lines,
    parse_probability,
    parse_probability_with_line,
)

REPORT = "city report: population 1234, pollution 30"

QUESTIONS = [
    {
        "question_id": f"bruce_disasters_seed42_T960_H240_A{i}",
        "question_text": f"Will event {i} happen by turn 1200?",
        "scenario_id": "bruce_disasters_seed42",
        "snapshot_turn": 960,
    }
    for i in range(1, 4)
]
LABELS = [f"test-model {q['question_id']}" for q in QUESTIONS]


def delimited(*lines: str) -> str:
    """A response in the requested format, with reasoning above the block."""
    return (
        "Considering the report, most of these events are unlikely.\n"
        "<<<PROBABILITIES>>>\n" + "\n".join(lines) + "\n<<<END>>>"
    )


class TestParseBatchProbabilities:
    def test_q_numbered_lines(self):
        response = delimited("Q1: 0.65", "Q2: 0.03", "Q3: 1")
        assert parse_batch_probabilities(response, LABELS) == [0.65, 0.03, 1.0]

    def test_dot_and_paren_numbered_lines(self):
        response = delimited("1. 0.65", "2) 0.03", "3: 0.9")
        assert parse_batch_probabilities(response, LABELS) == [0.65, 0.03, 0.9]

    def test_plain_lines_fall_back_to_positional(self):
        response = delimited("0.65", "0.03", "0.9")
        assert parse_batch_probabilities(response, LABELS) == [0.65, 0.03, 0.9]

    def test_comma_separated_answers_on_one_line(self):
        response = delimited("Q1: 0.65, Q2: 0.03, Q3: 0.9")
        assert parse_batch_probabilities(response, LABELS) == [0.65, 0.03, 0.9]

    def test_numbered_lines_ignore_order_and_skips(self):
        response = delimited("Q3: 0.9", "Q1: 0.65")
        assert parse_batch_probabilities(response, LABELS, quiet=True) == [
            0.65,
            None,
            0.9,
        ]

    def test_no_positional_fallback_after_a_partial_numbered_parse(self):
        # Q2's answer is prose; the bare line must not slide onto it.
        response = delimited("Q1: 0.65", "Q2: unlikely", "0.9")
        assert parse_batch_probabilities(response, LABELS, quiet=True) == [
            0.65,
            None,
            None,
        ]

    def test_restated_answer_wins(self):
        # Later lines overwrite earlier ones for the same number.
        response = delimited("Q1: 0.2", "Q2: 0.03", "Q1: 0.65")
        assert parse_batch_probabilities(response, LABELS, quiet=True) == [
            0.65,
            0.03,
            None,
        ]

    def test_trailing_prose_is_fine_leading_prose_is_not(self):
        response = delimited(
            "Q1: 0.65 (earthquakes are rare)",
            "Q2: fewer than 1 in 10",
            "Q3: 0.9",
        )
        assert parse_batch_probabilities(response, LABELS, quiet=True) == [
            0.65,
            None,
            0.9,
        ]

    def test_out_of_range_values_rejected_not_clamped(self):
        response = delimited("Q1: 1.5", "Q2: 65", "Q3: -1")
        assert parse_batch_probabilities(response, LABELS, quiet=True) == [
            None,
            None,
            None,
        ]

    def test_percent_values_divided_by_100(self):
        response = delimited("Q1: 65%", "Q2: 3 %", "Q3: 100%")
        assert parse_batch_probabilities(response, LABELS) == [0.65, 0.03, 1.0]

    def test_prose_numbers_are_not_answers(self):
        response = (
            "1. The population will likely fall.\n"
            "2. Roughly 5000 people could leave the city.\n"
        )
        assert parse_batch_probabilities(response, LABELS, quiet=True) == [
            None,
            None,
            None,
        ]

    def test_empty_response(self):
        assert parse_batch_probabilities(None, LABELS, quiet=True) == [None] * 3
        assert parse_batch_probabilities("", LABELS, quiet=True) == [None] * 3

    def test_no_block_still_parses_numbered_lines(self):
        response = "Q1: 0.65\nQ2: 0.03\nQ3: 0.9"
        assert parse_batch_probabilities(response, LABELS) == [0.65, 0.03, 0.9]

    def test_singular_probability_marker_accepted(self):
        response = "<<<PROBABILITY>>>\nQ1: 0.65\n<<<END>>>"
        assert parse_batch_probabilities(response, LABELS, quiet=True) == [
            0.65,
            None,
            None,
        ]

    def test_single_question_uses_single_format(self):
        assert parse_batch_probabilities(delimited("0.65"), LABELS[:1]) == [0.65]


class TestParseWithLines:
    """The line numbers are 1-based over the whole response, block included."""

    def test_numbered_lines(self):
        # delimited() puts one reasoning line and the marker above the block,
        # so its first answer line is line 3.
        response = delimited("Q1: 0.65", "Q2: 0.03", "Q3: 0.9")
        assert parse_batch_probabilities_with_lines(response, LABELS) == [
            (0.65, 3),
            (0.03, 4),
            (0.9, 5),
        ]

    def test_answers_sharing_a_line_report_it(self):
        response = delimited("Q1: 0.65, Q2: 0.03", "Q3: 0.9")
        assert parse_batch_probabilities_with_lines(response, LABELS) == [
            (0.65, 3),
            (0.03, 3),
            (0.9, 4),
        ]

    def test_out_of_order_and_missing(self):
        response = delimited("Q3: 0.9", "", "Q1: 0.65")
        assert parse_batch_probabilities_with_lines(response, LABELS, quiet=True) == [
            (0.65, 5),
            (None, None),
            (0.9, 3),
        ]

    def test_positional_fallback(self):
        response = delimited("0.65", "0.03", "0.9")
        assert parse_batch_probabilities_with_lines(response, LABELS) == [
            (0.65, 3),
            (0.03, 4),
            (0.9, 5),
        ]

    def test_rejected_value_has_no_line(self):
        response = delimited("Q1: 0.65", "Q2: 42", "Q3: 0.9")
        assert parse_batch_probabilities_with_lines(response, LABELS, quiet=True) == [
            (0.65, 3),
            (None, None),
            (0.9, 5),
        ]

    def test_no_block_counts_from_the_top(self):
        assert parse_probability_with_line("prose\n0.65") == (0.65, 2)
        assert parse_batch_probabilities_with_lines(
            "prose\nQ1: 0.65\nQ2: 0.03\nQ3: 0.9", LABELS
        ) == [(0.65, 2), (0.03, 3), (0.9, 4)]

    def test_single_question(self):
        assert parse_probability_with_line(delimited("0.65")) == (0.65, 3)
        assert parse_probability_with_line(delimited("", "Q1: 0.65")) == (0.65, 4)
        assert parse_probability_with_line(None, quiet=True) == (None, None)
        assert parse_batch_probabilities_with_lines(delimited("0.65"), LABELS[:1]) == [
            (0.65, 3)
        ]

    def test_empty_response(self):
        assert (
            parse_batch_probabilities_with_lines("", LABELS, quiet=True)
            == [(None, None)] * 3
        )


class TestParseProbability:
    def test_bare_and_prefixed_values(self):
        assert parse_probability(delimited("0.65")) == 0.65
        assert parse_probability(delimited("Q1: 0.65")) == 0.65
        assert parse_probability(delimited(".65")) == 0.65

    def test_out_of_range_rejected(self):
        assert parse_probability(delimited("42"), quiet=True) is None

    def test_empty_and_unparseable(self):
        assert parse_probability(None, quiet=True) is None
        assert parse_probability("no numbers here", quiet=True) is None


class TestExtractAnswerBlock:
    def test_default_marker_unchanged(self):
        response = "reasoning\n<<<PERCENTILES>>>\nQ1: p10=1\n<<<END>>>"
        assert _extract_answer_block(response) == "Q1: p10=1"

    def test_probability_marker_does_not_match_percentiles(self):
        response = "<<<PERCENTILES>>>\nQ1: 0.5\n<<<END>>> trailing"
        # No PROBABILITIES block: falls back to everything before <<<END>>>.
        block = _extract_answer_block(response, r"PROBABILIT(?:Y|IES)")
        assert "Q1: 0.5" in block

    def test_last_block_wins(self):
        # A reasoning model that restates the output format mid-thought emits a
        # placeholder block before the real one; the answers are in the last.
        response = (
            "Format reminder:\n"
            "<<<PROBABILITIES>>>\nQ1: 0.XX\n<<<END>>>\n"
            "Now the real answer.\n"
            "<<<PROBABILITIES>>>\nQ1: 0.65\n<<<END>>>"
        )
        block = _extract_answer_block(response, r"PROBABILIT(?:Y|IES)")
        assert block == "Q1: 0.65"
        assert parse_batch_probabilities(response, LABELS[:1]) == [0.65]


class TestBuildBatchPromptBinary:
    def test_prompt_structure(self):
        prompt = build_batch_prompt_binary(REPORT, QUESTIONS)
        assert prompt.count(REPORT) == 1
        assert "## Questions" in prompt
        for i, q in enumerate(QUESTIONS, 1):
            assert f"{i}. {q['question_text']}" in prompt
        assert "Return 3 numbered probabilities" in prompt
        assert "<<<PROBABILITIES>>>" in prompt
        assert "Q1: 0.5" in prompt
        assert "probability" in prompt


class TestBinaryPaths:
    def test_cache_lives_under_the_binary_root(self):
        path = binary_eval.batch_dir("bruce_disasters_seed42_T960")
        assert path.parts[-3:] == ("binary", "cache", "bruce_disasters_seed42_T960")

    def test_model_id_slugging(self):
        path = binary_eval.response_path("bid", "openai/gpt-4.1", "abc123")
        assert path.name == "response-openai_gpt-4.1-abc123.txt"
        usage = binary_eval.usage_path("bid", "openai/gpt-4.1", "abc123")
        assert usage.name == "usage-openai_gpt-4.1-abc123.json"
        suffixed = binary_eval.response_path("bid", "openai/gpt-4.1:lowef", "abc123")
        assert suffixed.name == "response-openai_gpt-4.1+lowef-abc123.txt"

    def test_data_path(self):
        assert binary_eval.data_path("x").parts[-3:] == ("binary", "x", "data.json")
