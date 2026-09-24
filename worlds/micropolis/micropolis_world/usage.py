"""Per-call token and cost accounting for the model calls this world makes.

prompt_model() is the one place micropolis talks to an LLM, so it is the one
place tokens and dollars can be read off a reply; this module holds the record
it produces and the LiteLLM-shaped reading of it.

Kept backend-free at import time — the one function that needs a client imports
llm_backend lazily — so the scoring and analysis scripts can read recorded usage
without pulling one in.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CallUsage:
    """Tokens and dollars for one completed LLM API call.

    cost_usd is None, never 0.0, when the backend couldn't price the call: an
    unknown price must never be silently reported as free. Anything
    summing these has to decide what to say about the unpriced calls, and a
    None forces that decision instead of hiding it.

    Attributes:
        model_id: The id we asked for, e.g. "google/gemini-2.5-pro".
        response_model: What the provider echoed back, which is often the bare
            name without the provider prefix, and sometimes a dated variant of
            the alias that was requested.
        input_tokens: Prompt tokens billed.
        output_tokens: Completion tokens billed, including reasoning tokens.
        total_tokens: Input plus output.
        reasoning_tokens: Of the output tokens, those spent thinking rather
            than answering. None when the provider doesn't report it.
        cached_input_tokens: Of the input tokens, those served from a provider
            prompt cache at a discount. None when not reported.
        cache_write_tokens: Input tokens billed at a premium to populate a
            provider prompt cache. None when not reported.
        cost_usd: What the call cost, or None if the model has no price entry.
        latency_ms: Wall-clock time spent in the API call.
        provider: Which upstream endpoint actually served the call, as the
            gateway reported it (e.g. "Google AI Studio"). None when the
            backend doesn't say — LiteLLM talks to one provider per model, so
            it never does. Recorded because a slug can be served by several
            providers at different quantizations and speeds, which shows up as
            otherwise unexplained variance between calls to "the same" model.
        finish_reason: Why the first choice stopped ("stop", "length", ...),
            as the provider reported it. Recorded so that a response with no
            answer in it can be told apart after the fact: a model that ran out
            of tokens and one that simply stopped talking look the same in the
            response file. None for a sidecar written before this was kept.
        response: Everything else the provider sent back, with the generated
            text removed — see response_metadata(). The typed fields above are
            the reading of it this module knows how to make; this is the raw
            record for whatever question comes up later (native finish reason,
            response id, system fingerprint, ...). None for older sidecars.
    """

    model_id: str
    response_model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    provider: str | None = None
    finish_reason: str | None = None
    response: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """The usage as plain JSON-serializable data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CallUsage":
        """Read back what to_dict() wrote, ignoring fields we don't know.

        Unknown keys are dropped rather than raising so that a record written
        by a later version of this dataclass still loads here.
        """
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def tokens(self) -> str:
        """The token counts as one phrase, for a progress line next to a call.

        Only input and output are always shown; the rest appear when the
        provider reported them as nonzero, so a plain call stays a short line
        and a reasoning or cache-hitting one says so.
        """
        parts = [f"{self.input_tokens} in", f"{self.output_tokens} out"]
        if self.reasoning_tokens:
            parts.append(f"{self.reasoning_tokens} reasoning")
        if self.cached_input_tokens:
            parts.append(f"{self.cached_input_tokens} cached")
        if self.cache_write_tokens:
            parts.append(f"{self.cache_write_tokens} cache write")
        return ", ".join(parts)

    def describe(self) -> str:
        """One-line human summary, cost first."""
        cost = "cost unknown" if self.cost_usd is None else f"${self.cost_usd:.4f}"
        return f"{cost}, {self.tokens()}"


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """One LLM reply: its text, why it stopped, and what it cost.

    text is None when the model produced no content — a reasoning model can
    spend its whole budget thinking and stop at finish_reason "length" with
    nothing written. That call is still billed, so usage is populated either
    way and a None text must not be read as a free call.

    Attributes:
        text: The reply, or None if the model wrote nothing.
        finish_reason: Why generation stopped ("stop", "length", ...).
        usage: Tokens and cost for the call.
        retries: Failed attempts before the one that returned this reply, so 0
            on a first-try success. Lives here rather than on CallUsage
            because it describes getting the answer, not what was billed for
            it: the usage sidecars record one paid call, while a retried call
            may have burned minutes of wall clock the latency never shows.
    """

    text: str | None
    finish_reason: str | None
    usage: CallUsage
    retries: int = 0


def save_usage(path: Path, usage: CallUsage) -> None:
    """Record one call's tokens and cost at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(usage.to_dict(), indent=2), encoding="utf-8")


def load_usage(path: Path) -> CallUsage | None:
    """The recorded usage at `path`, or None if it is absent or unreadable.

    Absent for every response cached before cost tracking existed, so a missing
    or malformed file is an ordinary "cost unknown", not an error.
    """
    if not path.exists():
        return None
    try:
        return CallUsage.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, TypeError):
        return None


def _detail(obj: Any, *names: str) -> int | None:
    """First present attribute among `names`, or None if obj has none of them.

    litellm's usage-detail wrappers delete their unset optional fields in
    __init__, so plain attribute access raises AttributeError rather than
    returning None, and the whole wrapper is None when nothing was reported.
    OpenRouter simply omits the key, which AttrDict also turns into an
    AttributeError. Field names also drift between litellm versions — cache
    writes were cache_creation_tokens before 1.96 and cache_write_tokens after
    — so more than one name is tried and a rename degrades a field to None
    rather than raising.
    """
    if obj is None:
        return None
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def cost_from_response(response: Any, model_id: str | None = None) -> float | None:
    """USD cost of a response, or None if the backend couldn't price it.

    Never raises: pricing a call must not be able to fail the call that was
    already paid for. An unpriced call reports None rather than 0.0.

    Under OpenRouter the cost is the amount actually billed, carried on the
    response, so None is rare; under LiteLLM it is a price-map estimate and
    None means the model has no entry.

    Args:
        response: The object completion() returned.
        model_id: The id we asked for. Passed on as an extra candidate name,
            which matters to LiteLLM when the provider echoes back a bare
            "gemini-2.5-pro" while the price map is keyed
            "gemini/gemini-2.5-pro". Ignored by the OpenRouter backend.

    Returns:
        The cost in USD, or None if the backend can't price the model.
    """
    # Where OpenRouter puts the billed cost, and where LiteLLM puts it behind a
    # proxy or on the batch path. Checked first: it's the only correct value
    # when it is there.
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict) and hidden.get("response_cost") is not None:
        try:
            return float(hidden["response_cost"])
        except (TypeError, ValueError):
            pass

    # Imported here so the scoring-only scripts don't pull in an LLM client.
    from .llm_backend import completion_cost

    try:
        cost = completion_cost(completion_response=response, model=model_id)
    except Exception:  # noqa: BLE001 - both backends raise when they can't price
        return None
    return None if cost is None else float(cost)


# The keys of a choice's message that hold generated text. Dropped from the
# recorded response: the answer is the response file beside the sidecar, and
# reasoning text can run to many KB per call. Everything else in the message
# (role, refusal, tool calls, ...) is metadata and stays.
GENERATED_TEXT_KEYS = ("content", "reasoning", "reasoning_content", "reasoning_details")


def response_metadata(response: Any) -> dict[str, Any] | None:
    """The response body as plain data, minus the text the model generated.

    Both backends can render themselves as a dict (model_dump); the
    OpenRouter one also tacks on the private _hidden_params it built for
    LiteLLM parity, which carries the whole request — the prompt, cached
    separately — and is not part of the response, so it is left out. Never
    raises: a response whose shape this cannot read records None rather than
    failing the call that was already paid for.
    """
    try:
        dump = getattr(response, "model_dump", None)
        data = dump() if callable(dump) else dict(response)
        data = json.loads(json.dumps(data, default=str))
    except Exception:  # noqa: BLE001 - recording metadata must never fail a call
        return None
    if not isinstance(data, dict):
        return None
    data.pop("_hidden_params", None)
    for choice in data.get("choices") or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(message, dict):
            for key in GENERATED_TEXT_KEYS:
                message.pop(key, None)
    return data


def usage_from_response(
    response: Any,
    model_id: str,
    latency_ms: float | None = None,
) -> CallUsage:
    """Tokens and cost for a completion response.

    Args:
        response: The object completion() returned.
        model_id: The id we asked for, recorded as CallUsage.model_id.
        latency_ms: Wall-clock time the call took, if measured.

    Returns:
        The call's usage. Fields the provider didn't report are None.
    """
    usage = getattr(response, "usage", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)

    # Empty string as well as absent reads as "not reported": OpenRouter's
    # `provider` is "" on the rare reply that omits it, and a blank would
    # otherwise become its own bucket in the per-model provider check.
    hidden = getattr(response, "_hidden_params", None)
    provider = (hidden or {}).get("provider") if isinstance(hidden, dict) else None

    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    # total_tokens can be absent or 0 unless the provider sent it, so fall back
    # to the sum rather than reporting a total that contradicts its own parts.
    total_tokens = getattr(usage, "total_tokens", 0) or 0

    choices = getattr(response, "choices", None) or []
    finish_reason = getattr(choices[0], "finish_reason", None) if choices else None

    return CallUsage(
        model_id=model_id,
        response_model=getattr(response, "model", None),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens or (input_tokens + output_tokens),
        reasoning_tokens=_detail(completion_details, "reasoning_tokens"),
        cached_input_tokens=_detail(prompt_details, "cached_tokens"),
        cache_write_tokens=_detail(
            prompt_details, "cache_write_tokens", "cache_creation_tokens"
        ),
        cost_usd=cost_from_response(response, model_id),
        latency_ms=latency_ms,
        provider=provider or None,
        finish_reason=finish_reason,
        response=response_metadata(response),
    )
