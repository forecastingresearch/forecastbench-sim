"""The request body openrouter_completion builds: spec chosen by slug, model id sent.

conftest blocks httpx, so _build is tested directly rather than through a call.
"""

import pytest

from micropolis_world import openrouter_completion as orc

MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture
def specs(monkeypatch):
    monkeypatch.setattr(
        orc,
        "MODELS",
        {
            "prov/m": orc.ModelSpec(endpoint="e"),
            "prov/m:lowef": orc.ModelSpec(endpoint="e", reasoning_effort="low"),
        },
    )


def test_suffixed_slug_selects_its_spec_but_sends_the_model_id(specs):
    plain = orc._build("prov/m", MSGS, {})
    low = orc._build("prov/m:lowef", MSGS, {})

    assert plain["model"] == low["model"] == "prov/m"
    assert "reasoning" not in plain
    assert low["reasoning"] == {"effort": "low"}
    assert low["provider"]["only"] == ["e"]


def test_suffixed_slug_without_a_spec_is_rejected(specs):
    """A suffix exists only to pick a spec; silently sending the base id would
    hide the typo behind provider defaults."""
    with pytest.raises(orc.BadRequestError, match="prov/m:hief"):
        orc._build("prov/m:hief", MSGS, {})


def test_unsuffixed_slug_without_a_spec_is_sent_as_is(specs):
    body = orc._build("other/x", MSGS, {})
    assert body["model"] == "other/x"
    assert "provider" not in body
