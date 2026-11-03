"""Unit tests for the natural-conditional prompts (micropolis_world/natcond_prompts.py)."""

import pytest

from micropolis_world.binary_questions import QUESTION_IDS, QUESTIONS
from micropolis_world.natcond_prompts import (
    PREAMBLE_PATH,
    STATEMENT,
    build_jobs,
    conversation,
    explicit_question,
    fact_text,
    report_from_prompt,
    turn2_text,
)

REPORT = "## Game report\nCity: kyoto\nDisasters: enabled\n\n"
PROMPT = "preamble text\n\n" + REPORT + "## Questions\n1. Will ...?\n"


def cell(cell_id, qa="A13", qb="A1", hb=144):
    sid, S = "kyoto_disasters_seed42", 960
    return {
        "cell_id": cell_id, "scenario_id": sid, "snapshot_turn": S, "horizon_a": 240, "horizon_b": hb,
        "qid_a": qa, "qid_b": qb, "block": "inferential",
        "question_id_a": f"{sid}_T{S}_H240_{qa}", "question_id_b": f"{sid}_T{S}_H{hb}_{qb}",
        "text_a": f"Question {qa} by turn 1200?",
    }


def test_every_question_has_a_statement():
    assert set(STATEMENT) == set(QUESTION_IDS)
    assert PREAMBLE_PATH.exists()


def test_report_section():
    assert report_from_prompt(PROMPT) == REPORT
    with pytest.raises(ValueError):
        report_from_prompt("no report here")


def test_fact_names_the_reveal_window_and_turns():
    assert fact_text(cell(0)) == "at least one earthquake was reported at some point between turn 961 and turn 1104."
    assert fact_text(cell(0, qb="A7")) == "the city's population at turn 1104 was lower than at turn 960."


def test_every_question_restates_with_explicit_turns():
    for q in QUESTIONS:
        text = explicit_question(q.text.replace("{HORIZON}", "1200"), 960)
        assert "current" not in text and " now" not in text and "960" in text or "at turn 1200" in text


def test_turn2_message():
    c = cell(0, qa="A4", qb="A5")
    c["text_a"] = "Will a monster be sighted between the current turn and turn 1200?"
    assert turn2_text(c) == (
        "Suppose you now learn one more thing about the game after turn 960: an airplane crashed at some point"
        " between turn 961 and turn 1104. You learn nothing else about what happened after turn 960.\n\n"
        "Given this, what is the probability that this question resolves YES: Will a monster be sighted between"
        " turn 960 and turn 1200? Anything from turn 961 through turn 1200 counts toward it, including turns 961–1104.\n\n"
        "Reason about what, if anything, this changes, and end your response with your updated probability in"
        " <probability> </probability> tags."
    )
    c["text_a"] = "Will the city's classification at turn 1200 be lower than it is now (e.g. Metropolis → Capital)?"
    t = turn2_text(c)
    assert "lower than it was at turn 960 (e.g." in t and "counts toward it" not in t


def test_jobs():
    # cells 0 and 1 share question A13 (two reveals): one turn-1 prompt for both
    cells = [cell(0), cell(1, qb="A7"), cell(2, qa="A15")]
    jobs = build_jobs(cells, {("kyoto_disasters_seed42", 960): REPORT}, "PREAMBLE\n")
    assert sorted(jobs) == ["t1", "t2"]
    assert [len(jobs[a]) for a in ("t1", "t2")] == [2, 3]
    assert jobs["t2"][0]["question_id"] == jobs["t2"][1]["question_id"] == jobs["t1"][0]["question_id"]
    t1 = jobs["t1"][0]
    assert t1["prefix"] == "PREAMBLE\n\n\n" + REPORT
    assert "## Question\nQuestion A13 by turn 1200?\n\nThink" in t1["tail"] and "revealed" not in t1["tail"]
    msgs = conversation(t1, "reply", jobs["t2"][0])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == "reply" and msgs[2]["content"].startswith("Suppose you now learn one more thing")
