"""Tests for reading tokens and cost off a model reply.

Fixtures are built from LiteLLM's own response types rather than hand-rolled
stubs, so a field that LiteLLM renames or stops populating fails a test here
instead of silently reporting None in a run. They also stand in for the
OpenRouter backend's AttrDict, which answers the same attribute reads. The
client is stubbed by replacing the name prompt_model binds, so nothing here
calls a model or needs a key.
"""

import pytest
from fbsim_core.evaluation.models import LiteLLMModel
from litellm.types.utils import Choices, Message, ModelResponse, Usage

import micropolis_world.module_globals as g
from micropolis_world import llm_backend
from micropolis_world.usage import CallUsage, cost_from_response, usage_from_response


def make_response(
    model: str = "gpt-4o",
    content: str | None = "hello",
    finish_reason: str = "stop",
    usage: Usage | None = None,
) -> ModelResponse:
    """A LiteLLM response of the shape completion() returns."""
    return ModelResponse(
        id="chatcmpl-test",
        choices=[
            Choices(
                finish_reason=finish_reason,
                index=0,
                message=Message(content=content, role="assistant"),
            )
        ],
        created=1700000000,
        model=model,
        object="chat.completion",
        usage=usage or Usage(prompt_tokens=100, completion_tokens=50),
    )


@pytest.fixture
def calls(monkeypatch):
    """Stub the backend's completion, recording the kwargs prompt_model sends.

    Patching the attribute on llm_backend works only because prompt_model does
    its `from .llm_backend import completion` inside the function body, so the
    name is resolved per call. Hoisting that import to module level would rebind
    it once at import time and these tests would start making real API calls —
    which is exactly what happens to fbsim-core's models.py, and why it isn't
    stubbed here.
    """
    recorded = []

    def fake_completion(**kwargs):
        recorded.append(kwargs)
        return make_response()

    monkeypatch.setattr(llm_backend, "completion", fake_completion)
    return recorded


# --- reading a response -----------------------------------------------------


def test_reads_every_reported_field():
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        completion_tokens_details={"reasoning_tokens": 40},
        prompt_tokens_details={"cached_tokens": 20},
    )
    got = usage_from_response(make_response(usage=usage), "openai/gpt-4o", 123.5)

    assert got.model_id == "openai/gpt-4o"
    assert got.response_model == "gpt-4o"
    assert got.input_tokens == 100
    assert got.output_tokens == 50
    assert got.reasoning_tokens == 40
    assert got.cached_input_tokens == 20
    assert got.latency_ms == 123.5


def test_missing_details_are_none_not_an_error():
    """The regression test for LiteLLM's deleted-when-unset detail fields.

    Usage(prompt_tokens=..., completion_tokens=...) leaves both detail wrappers
    None, and the wrappers themselves `del` their unset optional attributes, so
    plain attribute access raises AttributeError. This fails loudly if the
    getattr-based reads are ever replaced with direct access.
    """
    got = usage_from_response(make_response(), "openai/gpt-4o")

    assert got.reasoning_tokens is None
    assert got.cached_input_tokens is None
    assert got.cache_write_tokens is None
    assert got.latency_ms is None


def test_total_falls_back_to_the_sum():
    """LiteLLM leaves total_tokens at 0 unless the provider sends it."""
    assert usage_from_response(make_response(), "openai/gpt-4o").total_tokens == 150


def test_total_prefers_what_the_provider_reported():
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=175)
    got = usage_from_response(make_response(usage=usage), "openai/gpt-4o")
    assert got.total_tokens == 175


# --- pricing ----------------------------------------------------------------


def test_billed_cost_on_the_response_is_what_is_reported():
    """The OpenRouter path: the amount actually charged, carried on the reply.

    Also LiteLLM's proxy and batch paths, which populate the same field.
    """
    response = make_response()
    response._hidden_params = {"response_cost": 0.0123}
    assert cost_from_response(response, "openai/gpt-4o") == 0.0123


def test_openrouter_usage_cost_is_read_through_the_backend():
    """completion_cost reads usage.cost off an OpenRouter-shaped response."""
    resp = llm_backend.completion_cost(
        completion_response={"usage": {"cost": 0.0042}}, model="openai/gpt-4o"
    )
    assert resp == 0.0042


def test_unpriced_response_is_none_and_never_raises():
    """A reply the backend can't price reports None, and does not raise.

    None rather than 0.0 so an unknown price is never mistaken for a free call.
    Both backends raise here — LiteLLM for a model absent from its price map,
    OpenRouter for a response with no cost field — and both must come back None.
    """
    response = make_response(model="definitely-not-a-real-model")
    assert cost_from_response(response, "openai/definitely-not-a-real-model") is None


def test_truncation_warning_goes_to_stderr(capsys):
    """A warning must not land on stdout, where the report is written."""
    g.warn_if_truncated("openai/gpt-5", "length")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[warning]" in captured.err
    assert "openai/gpt-5" in captured.err


def test_no_truncation_warning_on_a_normal_finish(capsys):
    g.warn_if_truncated("openai/gpt-5", "stop")
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_records_the_serving_provider_from_hidden_params():
    """OpenRouter says which upstream endpoint answered; keep it."""
    response = make_response()
    response._hidden_params = {"provider": "Google AI Studio"}
    assert usage_from_response(response, "google/gemini-2.5-flash").provider == (
        "Google AI Studio"
    )


def test_provider_is_none_when_the_backend_does_not_report_one():
    """LiteLLM never does, and OpenRouter can send "" — both mean unknown."""
    assert usage_from_response(make_response(), "openai/gpt-4o").provider is None
    blank = make_response()
    blank._hidden_params = {"provider": ""}
    assert usage_from_response(blank, "openai/gpt-4o").provider is None


def test_provider_roundtrips_through_the_sidecar_record():
    usage = CallUsage(model_id="openai/gpt-4o", provider="OpenAI")
    assert CallUsage.from_dict(usage.to_dict()).provider == "OpenAI"


def test_unpriced_response_still_reports_its_tokens():
    response = make_response(model="definitely-not-a-real-model")
    got = usage_from_response(response, "openai/definitely-not-a-real-model")
    assert got.cost_usd is None
    assert got.input_tokens == 100
    assert got.output_tokens == 50


# --- CallUsage as a record --------------------------------------------------


def test_roundtrips_through_json_shaped_data():
    usage = CallUsage(model_id="openai/gpt-4o", input_tokens=1, cost_usd=0.5)
    assert CallUsage.from_dict(usage.to_dict()) == usage


def test_from_dict_ignores_fields_it_does_not_know():
    """A record written by a later version of CallUsage still loads."""
    got = CallUsage.from_dict({"model_id": "openai/gpt-4o", "some_future_field": 7})
    assert got.model_id == "openai/gpt-4o"


def test_describe_says_unknown_rather_than_zero():
    assert "cost unknown" in CallUsage(model_id="x").describe()
    assert "$0.0500" in CallUsage(model_id="x", cost_usd=0.05).describe()


def test_tokens_always_reports_input_and_output():
    got = CallUsage(model_id="x", input_tokens=2433, output_tokens=1638).tokens()
    assert got == "2433 in, 1638 out"


def test_tokens_omits_what_the_provider_reported_as_zero():
    """Anthropic sends explicit zeros for these; a plain call stays a short line."""
    got = CallUsage(
        model_id="x",
        input_tokens=1,
        output_tokens=2,
        reasoning_tokens=0,
        cached_input_tokens=0,
        cache_write_tokens=0,
    ).tokens()
    assert got == "1 in, 2 out"


def test_tokens_reports_reasoning_when_there_was_any():
    got = CallUsage(
        model_id="x", input_tokens=1, output_tokens=900, reasoning_tokens=850
    ).tokens()
    assert got == "1 in, 900 out, 850 reasoning"


def test_tokens_reports_cache_activity_when_there_was_any():
    got = CallUsage(
        model_id="x",
        input_tokens=5,
        output_tokens=2,
        cached_input_tokens=4000,
        cache_write_tokens=120,
    ).tokens()
    assert got == "5 in, 2 out, 4000 cached, 120 cache write"


# --- prompt_model ------------------------------------------------------------


def test_prompt_model_returns_text_finish_reason_and_usage(calls):
    got = g.prompt_model(LiteLLMModel("openai/gpt-4o"), "hi")

    assert got.text == "hello"
    assert got.finish_reason == "stop"
    assert got.usage.model_id == "openai/gpt-4o"
    assert got.usage.input_tokens == 100
    assert got.usage.output_tokens == 50
    assert got.usage.latency_ms is not None


def test_prompt_model_sends_the_slug_verbatim(calls):
    """The backend maps slug -> model id itself, so it must see the suffix."""
    g.prompt_model(LiteLLMModel("google/gemini-2.5-flash"), "hi")
    g.prompt_model(LiteLLMModel("openai/o3:lowef"), "hi")
    assert [c["model"] for c in calls] == ["google/gemini-2.5-flash", "openai/o3:lowef"]


def test_prompt_model_sends_no_sampling_params_or_token_cap(calls):
    """Every model runs on its provider defaults; the cap is the backend's."""
    g.prompt_model(LiteLLMModel("openai/gpt-4o"), "hi")
    g.prompt_model(LiteLLMModel("openai/gpt-5"), "hi")

    for call in calls:
        assert "temperature" not in call
        # The output cap belongs to model_specs.json5, not to a caller.
        assert "max_tokens" not in call


def test_prompt_model_reports_usage_for_an_empty_reply(monkeypatch):
    """A reasoning model that thinks past its budget writes nothing but bills."""

    def fake_completion(**kwargs):
        return make_response(
            content=None,
            finish_reason="length",
            usage=Usage(prompt_tokens=100, completion_tokens=4000),
        )

    monkeypatch.setattr(llm_backend, "completion", fake_completion)
    got = g.prompt_model(LiteLLMModel("openai/gpt-4o"), "hi")

    assert got.text is None
    assert got.finish_reason == "length"
    # The billed tokens are recorded even though nothing was written.
    assert got.usage.output_tokens == 4000


# --- the response's own metadata --------------------------------------------


def test_records_the_finish_reason():
    usage = usage_from_response(make_response(finish_reason="length"), "openai/gpt-4o")
    assert usage.finish_reason == "length"


def test_records_the_response_minus_the_generated_text():
    usage = usage_from_response(
        make_response(content="the whole answer", finish_reason="stop"), "openai/gpt-4o"
    )
    assert usage.response is not None
    assert usage.response["id"] == "chatcmpl-test"
    assert usage.response["object"] == "chat.completion"
    choice = usage.response["choices"][0]
    assert choice["finish_reason"] == "stop"
    # The text lives in the response file beside the sidecar, not here.
    assert "content" not in choice["message"]
    assert "the whole answer" not in str(usage.response)


def test_response_metadata_drops_the_backends_private_params():
    response = make_response()
    response._hidden_params = {"request_body": {"messages": ["the prompt"]}}
    usage = usage_from_response(response, "openai/gpt-4o")
    assert "_hidden_params" not in usage.response
    assert "the prompt" not in str(usage.response)


def test_response_metadata_is_json_data():
    import json

    usage = usage_from_response(make_response(), "openai/gpt-4o")
    json.dumps(usage.to_dict())  # must not raise


def test_a_response_of_unknown_shape_records_no_metadata_rather_than_failing():
    class Odd:
        usage = None
        choices = ()

    usage = usage_from_response(Odd(), "openai/gpt-4o")
    assert usage.response is None
    assert usage.finish_reason is None


def test_a_sidecar_written_before_metadata_was_kept_still_loads():
    old = {
        "model_id": "openai/gpt-4o",
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "cost_usd": 0.001,
    }
    usage = CallUsage.from_dict(old)
    assert usage.finish_reason is None
    assert usage.response is None
    assert usage.cost_usd == 0.001
