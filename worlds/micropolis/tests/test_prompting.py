"""Tests for the concurrent prompting fan-out.

The client is stubbed by replacing the name prompt_model_async binds — patching
llm_backend.acompletion works only because the import happens inside the
function body, exactly like prompt_model's (see test_usage.py's `calls`
fixture). Errors are raised as the backend's own classes, which is what the
retry loop catches whichever backend llm_backend has live. All tests drive the
async code with asyncio.run, so no async test plugin is needed and nothing here
calls a model or needs a key.
"""

import asyncio
import time

import pytest
from fbsim_core.evaluation.models import LiteLLMModel
from test_usage import make_response

from micropolis_world import llm_backend
from micropolis_world.prompting import (
    DEFAULT_CONCURRENCY,
    PromptJob,
    PromptResult,
    format_eta,
    format_latency,
    prompt_model_async,
    run_prompts,
)


@pytest.fixture
def calls(monkeypatch):
    """Stub the backend's acompletion, recording the kwargs each call sends."""
    recorded = []

    async def fake_acompletion(**kwargs):
        recorded.append(kwargs)
        return make_response()

    monkeypatch.setattr(llm_backend, "acompletion", fake_acompletion)
    return recorded


def job(model_id: str, content: str = "hi") -> PromptJob:
    return PromptJob(
        key=model_id,
        model=LiteLLMModel(model_id),
        model_name=model_id,
        messages=[{"role": "user", "content": content}],
    )


def collect(jobs, **kwargs) -> list[PromptResult]:
    async def consume():
        return [result async for result in run_prompts(jobs, **kwargs)]

    return asyncio.run(consume())


# --- format_eta ---------------------------------------------------------------


def test_format_eta_is_mean_time_per_completion_times_whats_left():
    start = time.perf_counter() - 10
    assert format_eta(start, 1, 3) == "  ~20s left"
    assert format_eta(start, 2, 3) == "  ~5s left"
    # Nothing left to wait for: empty, so callers can append unconditionally.
    assert format_eta(start, 3, 3) == ""


def test_format_eta_switches_to_minutes():
    start = time.perf_counter() - 120
    assert format_eta(start, 1, 3) == "  ~4m left"


# --- prompt_model_async -------------------------------------------------------


def test_returns_text_finish_reason_and_usage(calls):
    got = asyncio.run(
        prompt_model_async(
            LiteLLMModel("openai/gpt-4o"),
            [{"role": "user", "content": "hi"}],
        )
    )

    assert got.text == "hello"
    assert got.finish_reason == "stop"
    assert got.usage.model_id == "openai/gpt-4o"
    assert got.usage.input_tokens == 100
    assert got.usage.output_tokens == 50
    assert got.usage.latency_ms is not None


def test_sends_backend_id_and_no_sampling_params(calls):
    """The kwargs must match sync prompt_model's for the same model."""
    model_ids = [
        "google/gemini-2.5-flash",
        "openai/gpt-4o",
        "openai/gpt-5",
        "openai/gpt-5:lowef",
    ]
    for model_id in model_ids:
        asyncio.run(
            prompt_model_async(
                LiteLLMModel(model_id),
                [{"role": "user", "content": "hi"}],
            )
        )

    # The slug goes to the backend verbatim; mapping slug -> model id (and
    # picking the suffixed spec) is the backend's job.
    assert [c["model"] for c in calls] == model_ids
    for call in calls:
        assert "temperature" not in call
        # The output cap belongs to the backend's registry, not to a caller.
        assert "max_tokens" not in call


def test_owns_retries_and_disables_client_and_sdk_retries(calls):
    """Both silent retry layers must be off so our warning loop sees each 429."""
    asyncio.run(
        prompt_model_async(
            LiteLLMModel("openai/gpt-4o"),
            [{"role": "user", "content": "hi"}],
            num_retries=7,
            timeout=123.0,
        )
    )

    assert calls[0]["num_retries"] == 0
    assert calls[0]["max_retries"] == 0
    assert calls[0]["timeout"] == 123.0


def test_retries_rate_limits_with_a_stderr_warning(monkeypatch, capsys):
    attempts = []

    async def fake_acompletion(**kwargs):
        attempts.append(kwargs)
        if len(attempts) < 3:
            raise llm_backend.RateLimitError("slow down")
        return make_response()

    monkeypatch.setattr(llm_backend, "acompletion", fake_acompletion)
    monkeypatch.setattr("micropolis_world.prompting.RATE_LIMIT_BACKOFF_BASE_S", 0.0)

    got = asyncio.run(
        prompt_model_async(
            LiteLLMModel("openai/gpt-4o"),
            [{"role": "user", "content": "hi"}],
        )
    )

    assert got.text == "hello"
    assert len(attempts) == 3
    # Two failures before the reply, reported so the progress line can say so.
    assert got.retries == 2
    err = capsys.readouterr().err
    assert err.count("RateLimitError") == 2
    assert "openai/gpt-4o" in err


def test_raises_once_the_retry_budget_is_spent(monkeypatch, capsys):
    attempts = []

    async def fake_acompletion(**kwargs):
        attempts.append(kwargs)
        raise llm_backend.RateLimitError("slow down")

    monkeypatch.setattr(llm_backend, "acompletion", fake_acompletion)
    monkeypatch.setattr("micropolis_world.prompting.RATE_LIMIT_BACKOFF_BASE_S", 0.0)

    with pytest.raises(llm_backend.RateLimitError):
        asyncio.run(
            prompt_model_async(
                LiteLLMModel("openai/gpt-4o"),
                [{"role": "user", "content": "hi"}],
                num_retries=2,
            )
        )

    assert len(attempts) == 3  # 1 first try + 2 retries
    # Warnings fire per retry, not for the final failure — that one raises.
    assert capsys.readouterr().err.count("RateLimitError") == 2


def test_transient_errors_retry_but_client_errors_raise_immediately(
    monkeypatch, capsys
):
    attempts = []

    async def flaky_acompletion(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise llm_backend.InternalServerError("oops")
        return make_response()

    monkeypatch.setattr(llm_backend, "acompletion", flaky_acompletion)
    monkeypatch.setattr("micropolis_world.prompting.TRANSIENT_BACKOFF_S", 0.0)

    got = asyncio.run(
        prompt_model_async(
            LiteLLMModel("openai/gpt-4o"),
            [{"role": "user", "content": "hi"}],
        )
    )
    assert got.text == "hello"
    assert len(attempts) == 2
    assert "InternalServerError" in capsys.readouterr().err

    async def unauthorized_acompletion(**kwargs):
        attempts.append(kwargs)
        raise llm_backend.AuthenticationError("bad key")

    monkeypatch.setattr(llm_backend, "acompletion", unauthorized_acompletion)
    attempts.clear()

    with pytest.raises(llm_backend.AuthenticationError):
        asyncio.run(
            prompt_model_async(
                LiteLLMModel("openai/gpt-4o"),
                [{"role": "user", "content": "hi"}],
            )
        )
    assert len(attempts) == 1  # no retry budget wasted on a hopeless call


def test_passes_the_message_list_through_verbatim(calls):
    """Multi-turn readiness: whatever conversation is given is what is sent."""
    conversation = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "and again"},
    ]
    asyncio.run(prompt_model_async(LiteLLMModel("openai/gpt-4o"), conversation))
    assert calls[0]["messages"] == conversation


# --- run_prompts --------------------------------------------------------------


def test_caps_in_flight_calls_globally_across_providers(monkeypatch):
    in_flight = 0
    max_seen = 0

    async def fake_acompletion(**kwargs):
        nonlocal in_flight, max_seen
        in_flight += 1
        max_seen = max(max_seen, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return make_response()

    monkeypatch.setattr(llm_backend, "acompletion", fake_acompletion)

    jobs = [job("openai/gpt-4o", content=f"q{i}") for i in range(4)] + [
        job("anthropic/claude-sonnet-4-5", content=f"q{i}") for i in range(4)
    ]
    results = collect(jobs, limit=2)

    assert len(results) == len(jobs)
    # The cap is one global budget, not per provider: mixing providers must not
    # multiply how many calls are in flight at once.
    assert max_seen <= 2


def test_default_concurrency_applies_when_no_limit_given(monkeypatch):
    in_flight = 0
    max_seen = 0

    async def fake_acompletion(**kwargs):
        nonlocal in_flight, max_seen
        in_flight += 1
        max_seen = max(max_seen, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return make_response()

    monkeypatch.setattr(llm_backend, "acompletion", fake_acompletion)

    jobs = [
        job("openai/gpt-4o", content=f"q{i}") for i in range(DEFAULT_CONCURRENCY + 6)
    ]
    results = collect(jobs)

    assert len(results) == len(jobs)
    assert max_seen <= DEFAULT_CONCURRENCY


def test_yields_in_completion_order_each_job_once(monkeypatch):
    async def fake_acompletion(**kwargs):
        await asyncio.sleep(0.05 if kwargs["model"].startswith("openai") else 0)
        return make_response()

    monkeypatch.setattr(llm_backend, "acompletion", fake_acompletion)

    jobs = [job("openai/gpt-4o"), job("mistral/mistral-large")]
    results = collect(jobs)

    assert [r.job.key for r in results] == ["mistral/mistral-large", "openai/gpt-4o"]


def test_captures_failures_per_job_and_continues(monkeypatch):
    async def fake_acompletion(**kwargs):
        if kwargs["model"].startswith("xai"):
            raise ValueError("no such model")
        return make_response()

    monkeypatch.setattr(llm_backend, "acompletion", fake_acompletion)

    results = collect([job("openai/gpt-4o"), job("xai/grok-4")])
    by_key = {r.job.key: r for r in results}

    failed = by_key["xai/grok-4"]
    assert not failed.ok
    assert failed.response is None
    assert isinstance(failed.error, ValueError)

    succeeded = by_key["openai/gpt-4o"]
    assert succeeded.ok
    assert succeeded.error is None
    assert succeeded.response.text == "hello"


def test_first_try_success_reports_no_retries(calls):
    got = asyncio.run(
        prompt_model_async(
            LiteLLMModel("openai/gpt-4o"),
            [{"role": "user", "content": "hi"}],
        )
    )

    assert got.retries == 0


def test_format_latency_names_retries_only_when_there_were_some():
    assert format_latency(8321.4) == ", 8.3s"
    assert format_latency(8321.4, 0) == ", 8.3s"
    assert format_latency(8321.4, 1) == ", 8.3s 1 RETRY"
    assert format_latency(8321.4, 3) == ", 8.3s 3 RETRIES"
    # No recorded latency: nothing to append, retries or not. Every cached
    # response gathered before latency tracking lands here.
    assert format_latency(None) == ""
    assert format_latency(None, 2) == ""
