"""Concurrent LLM prompting: global rate limiting and async fan-out.

The scripts that prompt many models build a list of PromptJobs and consume
run_prompts() with `async for`, so the network calls overlap while every disk
write and print happens serially in the consumer — no locks, no interleaved
output.

Kept backend-free at import time, like usage.py: the one function that needs
an LLM client imports llm_backend lazily, so scoring-only code can import this
module's dataclasses without pulling one in.

prompt_model_async takes a full message list rather than a prompt string, so a
future multi-turn caller just appends assistant/user turns and calls it again;
run_prompts is only the single-shot fan-out convenience built on top of it and
ConcurrencyLimiter, both of which are public for exactly that reason.
"""

import asyncio
import random
import time
from collections.abc import AsyncIterator, Hashable
from dataclasses import dataclass
from typing import Any

from . import messages as msg
from .usage import LLMResponse, usage_from_response

# Max concurrent in-flight calls, across every model. One global cap rather
# than a per-provider one because every call leaves through OpenRouter: the
# binding limit is the gateway's in-flight budget (the dollar value of all
# simultaneously open requests on the account), which a per-provider map
# cannot express — its caps summed to their total, and exhausting that budget
# fails a call with an HTTP 402 the retry loop does not catch. A config's
# "concurrency" key overrides this.
DEFAULT_CONCURRENCY = 16

# Retries after the first attempt. prompt_model_async owns the retry loop:
# litellm's own retry paths (its tenacity wrapper and the provider SDK's
# max_retries) are completely silent, so a run deep in 429 territory just
# looked slow. We disable both layers and back off ourselves, printing a
# warning to stderr before each retry.
DEFAULT_NUM_RETRIES = 3
# Backoff schedule: exponential for 429s, a short constant pause for other
# transient errors (connection drops, 5xx, timeouts). Module-level so tests
# can zero them out.
RATE_LIMIT_BACKOFF_BASE_S = 1.0
RATE_LIMIT_BACKOFF_MAX_S = 30.0
TRANSIENT_BACKOFF_S = 1.0
# Per-attempt cap, forwarded to the provider client. Generous: a 60k-token
# reasoning reply can legitimately run many minutes.
DEFAULT_TIMEOUT_S = 900.0


def format_eta(start: float, done: int, total: int) -> str:
    """Crude time-left note: mean seconds per completion times what is left.

    `start` is the time.perf_counter() reading from just before the fan-out.
    Deliberately simple — it knows nothing about which calls are still in
    flight or how providers differ in speed, just enough to answer "wait or
    walk away?". Returns "" once everything is done, and carries its own
    leading separator, so callers can append it to a line unconditionally.
    """
    if done >= total:
        return ""
    seconds = (time.perf_counter() - start) / done * (total - done)
    if seconds >= 60:
        return f"  ~{seconds / 60:.0f}m left"
    return f"  ~{seconds:.0f}s left"


def format_latency(latency_ms: float | None, retries: int = 0) -> str:
    """How long one API call took, as a phrase for a progress line.

    Reads CallUsage.latency_ms, which prompt_model_async measures around the
    acompletion await — the time from the request going out to the reply
    landing, for the attempt that succeeded, not the retries before it. Those
    are reported separately, as "2 RETRIES" after the time, because they are
    what explains a call that took far longer in wall clock than the time
    shown; a first-try success says nothing at all.

    Returns "" when the latency was never recorded, and carries its own
    leading separator, so callers can append it unconditionally.
    """
    if latency_ms is None:
        return ""
    took = f", {latency_ms / 1000:.1f}s"
    if retries:
        took += f" {retries} RETR{'Y' if retries == 1 else 'IES'}"
    return took


class ConcurrencyLimiter:
    """One global cap on in-flight calls, as a lazily created asyncio.Semaphore.

    Every model shares the single semaphore, because every call leaves through
    the same gateway and it is the gateway's in-flight budget that binds. The
    semaphore is created on first use because it must be instantiated inside
    the running event loop.
    """

    def __init__(self, limit: int | None = None):
        self._limit = DEFAULT_CONCURRENCY if limit is None else limit
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def limit(self) -> int:
        """The concurrency cap."""
        return self._limit

    def semaphore(self) -> asyncio.Semaphore:
        """The (lazily created) semaphore shared by every call."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._limit)
        return self._semaphore

    def __repr__(self) -> str:
        return f"ConcurrencyLimiter(limit={self._limit})"


async def prompt_model_async(
    model: Any,
    messages: list[dict],
    *,
    num_retries: int = DEFAULT_NUM_RETRIES,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> LLMResponse:
    """Send `messages` to `model`, returning its text, finish reason and cost.

    The async twin of module_globals.prompt_model, and kept in step with it:
    same normalized model id, same sampling defaults, same LLMResponse. The
    kwargs logic is knowingly duplicated rather than shared — factoring it out
    would couple the two lazy-import boundaries the tests monkeypatch through.
    The differences: a full message list instead of a single user prompt (so a
    multi-turn caller can pass a running conversation), and retry/timeout
    settings, which matter once calls run concurrently and a provider starts
    returning 429s.

    Retries happen here, not in the client: num_retries=0 and max_retries=0
    disable LiteLLM's wrapper retries and the provider SDK's, both of which
    are silent (the OpenRouter backend consumes both kwargs and has no retry
    layer of its own). Rate limits back off exponentially, other transient
    errors (connection drops, 5xx, timeouts) pause briefly, and each retry
    prints a one-line warning to stderr so a throttled run is visible.
    Anything else (auth failures, bad requests) raises immediately.

    The output cap is not set here: it is a per-endpoint limit, so it belongs
    to the backend's model registry rather than to a caller.

    `model` is duck-typed (.id) so importing this module never imports
    fbsim-core.
    """
    # Imported here so tests can monkeypatch the backend's acompletion; the
    # name is resolved per call. See test_prompting.py's `calls` fixture for
    # why hoisting this to module level would break the stub.
    from .llm_backend import (
        APIConnectionError,
        InternalServerError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout,
        acompletion,
    )

    kwargs = {
        "model": model.id,  # the slug; the backend maps it to its own id
        "messages": messages,
        "num_retries": 0,
        "max_retries": 0,
        "timeout": timeout,
    }

    for attempt in range(num_retries + 1):
        start = time.perf_counter()
        try:
            response = await acompletion(**kwargs)
        except (
            RateLimitError,
            APIConnectionError,
            InternalServerError,
            ServiceUnavailableError,
            Timeout,
        ) as e:
            if attempt == num_retries:
                raise
            if isinstance(e, RateLimitError):
                delay = min(
                    RATE_LIMIT_BACKOFF_BASE_S * 2**attempt, RATE_LIMIT_BACKOFF_MAX_S
                )
            else:
                delay = TRANSIENT_BACKOFF_S
            msg.retry(
                f"{model.id}: {type(e).__name__} on attempt"
                f" {attempt + 1}/{num_retries + 1}, retrying in {delay:.0f}s"
            )
            await asyncio.sleep(delay)
        else:
            break
    latency_ms = (time.perf_counter() - start) * 1000

    choice = response.choices[0]  # type: ignore
    return LLMResponse(
        text=choice.message.content,
        finish_reason=choice.finish_reason,
        usage=usage_from_response(response, model.id, latency_ms),
        # `attempt` is the index of the try that succeeded, so it counts the
        # failed ones before it: 0 on a first-try success.
        retries=attempt,
    )


@dataclass(frozen=True)
class PromptJob:
    """One model call to make: who to ask, what to send, how to name it back.

    key is an opaque caller identity — e.g. (batch_id, model_name) — echoed
    back on the PromptResult so the consumer can route the reply without
    keeping its own bookkeeping. model_name is the id the caller asked for
    (the config spelling), carried separately because model is duck-typed.
    """

    key: Hashable
    model: Any
    model_name: str
    messages: list[dict]


@dataclass(frozen=True)
class PromptResult:
    """One finished PromptJob: a response, or the exception that ended it.

    Exactly one of response/error is set. Errors are carried, not raised, so
    one failed call never tears down the rest of a fan-out; the consumer
    decides what a failure costs.
    """

    job: PromptJob
    response: LLMResponse | None
    error: Exception | None

    @property
    def ok(self) -> bool:
        return self.error is None


async def run_prompts(
    jobs: list[PromptJob],
    *,
    limit: int | None = None,
    num_retries: int = DEFAULT_NUM_RETRIES,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> AsyncIterator[PromptResult]:
    """Run every job concurrently, yielding results in completion order.

    At most `limit` calls (default DEFAULT_CONCURRENCY) are ever in flight,
    counted across every model. The semaphore is held across
    prompt_model_async's retries, so a throttled run stays throttled while it
    backs off rather than being hit by the next waiting call.

    Yields one PromptResult per job as each finishes. Only Exception is
    captured into results — CancelledError and KeyboardInterrupt propagate, so
    Ctrl-C still stops the run; the finally block then cancels whatever is
    still in flight.
    """
    limiter = ConcurrencyLimiter(limit)
    # We shuffle the list of jobs, so we get better time estimates sooner
    random.shuffle(jobs)

    async def worker(job: PromptJob) -> PromptResult:
        async with limiter.semaphore():
            try:
                response = await prompt_model_async(
                    job.model,
                    job.messages,
                    num_retries=num_retries,
                    timeout=timeout,
                )
                return PromptResult(job, response, None)
            except Exception as e:  # noqa: BLE001 - carried to the consumer
                return PromptResult(job, None, e)

    tasks = [asyncio.create_task(worker(job)) for job in jobs]
    try:
        for future in asyncio.as_completed(tasks):
            yield await future
    finally:
        # Early exit (an exception in the consumer, Ctrl-C) must not leave
        # workers running against a closing event loop.
        for task in tasks:
            task.cancel()
