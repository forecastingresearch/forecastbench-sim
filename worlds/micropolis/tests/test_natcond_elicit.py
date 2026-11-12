"""Unit tests for the natural-conditional elicitation (micropolis_world/natcond_elicit.py); no model calls."""

import asyncio

import pytest

import micropolis_world.natcond_elicit as ne
from micropolis_world.natcond_elicit import Results, elicit, parse_prob
from micropolis_world.prompting import PromptResult
from micropolis_world.usage import CallUsage, LLMResponse


@pytest.mark.parametrize(
    "text, want",
    [
        ("... <probability>0.35</probability>", (0.35, "strict")),
        ("<probability>0.2</probability> then <probability>0.4</probability>", (0.4, "strict")),
        ("<probability>35%</probability>", (0.35, "strict")),
        ("so my final probability is 0.12", (0.12, "lenient")),
        ("No number here.", (None, None)),
        (None, (None, None)),
    ],
)
def test_parse_prob(text, want):
    assert parse_prob(text) == want


JOBS = {
    "t1": [{"arm": "t1", "item": "q1", "question_id": "q1", "prefix": "P1 ", "tail": "ask q1"},
           {"arm": "t1", "item": "q2", "question_id": "q2", "prefix": "P2 ", "tail": "ask q2"}],
    "t2": [{"arm": "t2", "item": "q1|b1", "question_id": "q1", "cell_id": 0, "block": "mechanical", "followup": "fact b1"},
           {"arm": "t2", "item": "q1|b2", "question_id": "q1", "cell_id": 1, "block": "irrelevant", "followup": "fact b2"},
           {"arm": "t2", "item": "q2|b1", "question_id": "q2", "cell_id": 2, "block": "inferential", "followup": "fact b1"}],
}


def fake_backend(monkeypatch, answer):
    """Replace run_prompts: answer(messages, call_no) -> reply text; records every job sent."""
    sent = []

    async def run_prompts(todo, limit=None, num_retries=None):
        for job in todo:
            sent.append(job)
            text = answer(job.messages, len(sent))
            yield PromptResult(job, LLMResponse(text, "stop", CallUsage(model_id=job.model_name, cost_usd=0.001)), None)

    monkeypatch.setattr(ne, "run_prompts", run_prompts)
    return sent


def test_turn2_continues_own_reply_and_resumes(tmp_path, monkeypatch):
    sent = fake_backend(monkeypatch, lambda msgs, i: f"reply{i} <probability>0.{i}</probability>")
    res = Results(tmp_path / "results.jsonl")
    asyncio.run(elicit(JOBS, ["m/a"], res, log=lambda *_: None))
    assert [j.key[1] for j in sent] == ["t1", "t1", "t2", "t2", "t2"]
    t2 = next(j for j in sent if j.key == ("m/a", "t2", "q1|b2"))
    t1_reply = res.best(("m/a", "t1", "q1"))["text"]
    assert [m["role"] for m in t2.messages] == ["user", "assistant", "user"]
    assert t2.messages[0]["content"] == "P1 ask q1"
    assert t2.messages[1]["content"] == t1_reply and t2.messages[2]["content"] == "fact b2"
    assert res.best(("m/a", "t2", "q1|b2"))["t1_value"] == res.best(("m/a", "t1", "q1"))["value"]
    # a rerun from the file sends nothing
    sent.clear()
    asyncio.run(elicit(JOBS, ["m/a"], Results(tmp_path / "results.jsonl"), log=lambda *_: None))
    assert sent == []


CHANGED = {"t1": JOBS["t1"], "t2": [dict(JOBS["t2"][0], followup="new wording")] + JOBS["t2"][1:]}


def test_changed_turn2_prompt_is_asked_again(tmp_path, monkeypatch):
    sent = fake_backend(monkeypatch, lambda msgs, i: f"<probability>0.{i}</probability>")
    assert asyncio.run(elicit(JOBS, ["m/a"], Results(tmp_path / "results.jsonl"), log=lambda *_: None)) == (5, 5)
    sent.clear()
    res = Results(tmp_path / "results.jsonl")
    assert asyncio.run(elicit(CHANGED, ["m/a"], res, log=lambda *_: None)) == (5, 5)
    assert [j.key for j in sent] == [("m/a", "t2", "q1|b1")]  # turn 1 and the other cells reused
    assert len(res.rows["m/a", "t2", "q1|b1"]) == 2  # the old row stays on file


def test_answer_to_an_old_prompt_is_not_counted(tmp_path, monkeypatch):
    fake_backend(monkeypatch, lambda msgs, i: f"<probability>0.{i}</probability>")
    asyncio.run(elicit(JOBS, ["m/a"], Results(tmp_path / "results.jsonl"), log=lambda *_: None))
    fake_backend(monkeypatch, lambda msgs, i: "no number at all.")
    res = Results(tmp_path / "results.jsonl")
    assert asyncio.run(elicit(CHANGED, ["m/a"], res, reask=0, log=lambda *_: None)) == (4, 5)


def test_unparsed_reply_is_reasked_once(tmp_path, monkeypatch):
    sent = fake_backend(monkeypatch, lambda msgs, i: "no number at all.")
    res = Results(tmp_path / "results.jsonl")
    asyncio.run(elicit({"t1": JOBS["t1"][:1], "t2": []}, ["m/a"], res, reask=1, log=lambda *_: None))
    assert len(sent) == 2 and len(res.rows["m/a", "t1", "q1"]) == 2
    assert not res.needs(("m/a", "t1", "q1"), reask=1)


def test_no_turn2_without_a_turn1_reply(tmp_path, monkeypatch):
    sent = fake_backend(monkeypatch, lambda msgs, i: None if len(msgs) == 1 else "<probability>0.5</probability>")
    res = Results(tmp_path / "results.jsonl")
    asyncio.run(elicit(JOBS, ["m/a"], res, reask=0, log=lambda *_: None))
    assert all(j.key[1] == "t1" for j in sent)
