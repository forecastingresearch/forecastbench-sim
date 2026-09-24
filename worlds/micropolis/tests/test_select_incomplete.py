"""Tests for select_for_config's --incomplete path."""

from pathlib import Path

import pytest

from micropolis_world.config import Config
from micropolis_world.continuous_eval import (
    DatasetError,
    Response,
    ResponseId,
    select_for_config,
)

SEED = 42
CITIES = ["bruce", "kyoto"]
MODELS = ["m/a", "m/b"]

CFG = Config(
    {
        "models": MODELS,
        "cities": CITIES,
        "disasters": [False],
        "snapshot_turns": [960],
        "horizons": [144],
    },
    Path("test.json5"),
)


def make_corpus(scenario_ids: list[str]) -> list[dict]:
    return [
        {
            "question_id": f"{sid}_T960_H144_cityPop",
            "scenario_id": sid,
            "snapshot_turn": 960,
            "horizon": 144,
        }
        for sid in scenario_ids
    ]


def scenario_ids() -> list[str]:
    from micropolis_world.continuous_eval import scenario_ids_from

    return scenario_ids_from(CFG, SEED)


def responses_for(corpus, models) -> dict:
    return {
        ResponseId(m, c["question_id"]): Response(
            actual=1.0, percentiles=None, response_text=""
        )
        for c in corpus
        for m in models
    }


class TestIncomplete:
    def test_complete_dataset_is_unaffected(self):
        corpus = make_corpus(scenario_ids())
        resp = responses_for(corpus, MODELS)
        for flag in (False, True):
            sel_c, _, sel_m = select_for_config(
                corpus, resp, MODELS, CFG, SEED, incomplete=flag
            )
            assert len(sel_c) == len(corpus)
            assert sel_m == MODELS

    def test_missing_row_errors_without_the_flag(self):
        corpus = make_corpus(scenario_ids())
        resp = responses_for(corpus, MODELS)
        del resp[ResponseId("m/b", corpus[0]["question_id"])]
        with pytest.raises(DatasetError, match="missing 1 forecast"):
            select_for_config(corpus, resp, MODELS, CFG, SEED)

    def test_incomplete_keeps_every_gathered_pair(self, capsys):
        # The selection is ragged: the question m/b lacks is still scored for
        # m/a, and the corpus keeps it.
        corpus = make_corpus(scenario_ids())
        resp = responses_for(corpus, MODELS)
        hole = corpus[0]["question_id"]
        del resp[ResponseId("m/b", hole)]
        sel_c, sel_r, sel_m = select_for_config(
            corpus, resp, MODELS, CFG, SEED, incomplete=True
        )
        assert [c["question_id"] for c in sel_c] == [c["question_id"] for c in corpus]
        assert sel_m == MODELS
        assert ResponseId("m/a", hole) in sel_r
        assert ResponseId("m/b", hole) not in sel_r
        assert len(sel_r) == len(corpus) * len(MODELS) - 1
        err = capsys.readouterr().err
        assert "--incomplete" in err
        # The per-model coverage line the user asked to keep.
        assert f"m/b: {len(corpus) - 1} of {len(corpus)}" in err

    def test_incomplete_errors_on_a_model_with_nothing(self):
        # A model with no forecast at all would be an empty column in every
        # figure, so it stays an error even under --incomplete.
        corpus = make_corpus(scenario_ids())
        resp = responses_for(corpus, MODELS)
        for c in corpus:
            del resp[ResponseId("m/b", c["question_id"])]
        with pytest.raises(DatasetError, match="no forecast at all"):
            select_for_config(corpus, resp, MODELS, CFG, SEED, incomplete=True)

    def test_incomplete_does_not_relax_coverage(self):
        # A model the dataset knows nothing about stays an error: that is a
        # config/dataset mismatch, not sparse data.
        corpus = make_corpus(scenario_ids())
        resp = responses_for(corpus, MODELS)
        with pytest.raises(DatasetError, match="does not cover"):
            select_for_config(
                corpus, resp, MODELS, CFG, SEED, models=["m/nope"], incomplete=True
            )
