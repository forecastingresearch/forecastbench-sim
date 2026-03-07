"""Per-run spend tracking for LiteLLM calls."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm
from litellm import completion_cost
from litellm.integrations.custom_logger import CustomLogger


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    number = _coerce_float(value)
    if number is None:
        return None
    return int(number)


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(exclude_none=False)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            dumped = value.dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            dumped = dict(value.__dict__)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    return {"value": str(value)}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _nested_get(data: dict[str, Any], *path: str) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _extract_usage(response_obj: Any) -> dict[str, Any]:
    usage: Any = None
    if isinstance(response_obj, dict):
        usage = response_obj.get("usage")
    else:
        usage = getattr(response_obj, "usage", None)
    return _json_safe(_to_plain_dict(usage))


def _extract_reasoning_tokens(usage: dict[str, Any]) -> int | None:
    candidates = [
        usage.get("reasoning_tokens"),
        _nested_get(usage, "completion_tokens_details", "reasoning_tokens"),
        _nested_get(usage, "output_tokens_details", "reasoning_tokens"),
        _nested_get(usage, "output_token_details", "reasoning_tokens"),
        _nested_get(usage, "completion_tokens_details", "reasoning"),
    ]
    for value in candidates:
        token_count = _coerce_int(value)
        if token_count is not None:
            return token_count
    return None


def _extract_duration_ms(start_time: Any, end_time: Any) -> float | None:
    if start_time is None or end_time is None:
        return None
    try:
        if hasattr(start_time, "timestamp") and hasattr(end_time, "timestamp"):
            return max(0.0, (end_time.timestamp() - start_time.timestamp()) * 1000.0)
        return max(0.0, (float(end_time) - float(start_time)) * 1000.0)
    except Exception:
        return None


def _with_provider_prefix(model: str, provider: str | None) -> str:
    if "/" in model:
        return model
    if provider:
        return f"{provider}/{model}"
    return model


def _extract_model_name(kwargs: dict[str, Any], response_obj: Any) -> str:
    provider = kwargs.get("custom_llm_provider")
    if not isinstance(provider, str) or not provider:
        provider = None

    model = kwargs.get("model")
    if isinstance(model, str) and model:
        return _with_provider_prefix(model, provider)
    response_model = None
    if isinstance(response_obj, dict):
        response_model = response_obj.get("model")
    else:
        response_model = getattr(response_obj, "model", None)
    if isinstance(response_model, str) and response_model:
        return _with_provider_prefix(response_model, provider)
    return "unknown"


def _base_model_name(model_name: str) -> str:
    name = model_name.split("/", 1)[-1].lower()
    if "-202" in name:
        name = name.rsplit("-202", 1)[0]
    return name


def _is_reasoning_model(model_name: str) -> bool:
    base = _base_model_name(model_name)
    if "reasoning" in base:
        return True
    if base.startswith(("o1", "o3", "o4", "gpt-5", "gpt-5.1", "gpt-5.2")):
        return True
    return False


def _extract_error_text(response_obj: Any) -> str | None:
    if response_obj is None:
        return None
    if isinstance(response_obj, BaseException):
        return str(response_obj)
    if isinstance(response_obj, dict):
        for key in ("error", "message", "detail"):
            if key in response_obj:
                return str(response_obj.get(key))
        return str(response_obj)
    return str(response_obj)


class RunSpendTracker(CustomLogger):
    """Custom LiteLLM callback that writes spend and usage events for a run."""

    def __init__(self, run_id: str, log_dir: Path):
        super().__init__()
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.log_dir / "spend_events.jsonl"
        self.summary_file = self.log_dir / "spend_summary.json"
        self._lock = threading.Lock()
        self._finalized = False
        self._event_count = 0
        self._seen_event_keys: set[tuple[str, str]] = set()
        self._summary_cache: dict[str, Any] | None = None
        self._stats: dict[str, Any] = {
            "run_id": run_id,
            "start_time": _utc_now_iso(),
            "success_calls": 0,
            "failure_calls": 0,
            "events_logged": 0,
            "cost_total_usd": 0.0,
            "cost_missing_calls": 0,
            "usage_missing_calls": 0,
            "prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "total_tokens_total": 0,
            "reasoning_tokens_total": 0,
            "reasoning_model_calls": 0,
            "reasoning_tokens_reported_calls": 0,
            "reasoning_tokens_missing_calls": 0,
            "reasoning_tokens_zero_calls": 0,
            "reasoning_tokens_positive_calls": 0,
            "per_model": {},
        }
        self.events_file.write_text("", encoding="utf-8")

    def _record_event(
        self,
        event_type: str,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        model = _extract_model_name(kwargs, response_obj)
        litellm_call_id = kwargs.get("litellm_call_id")
        if litellm_call_id is not None:
            litellm_call_id = str(litellm_call_id)
            if not litellm_call_id:
                litellm_call_id = None

        usage = _extract_usage(response_obj)
        prompt_tokens = _coerce_int(usage.get("prompt_tokens"))
        if prompt_tokens is None:
            prompt_tokens = _coerce_int(usage.get("input_tokens"))

        completion_tokens = _coerce_int(usage.get("completion_tokens"))
        if completion_tokens is None:
            completion_tokens = _coerce_int(usage.get("output_tokens"))

        total_tokens = _coerce_int(usage.get("total_tokens"))
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        reasoning_tokens = _extract_reasoning_tokens(usage)
        is_reasoning = _is_reasoning_model(model)
        if is_reasoning:
            if reasoning_tokens is None:
                reasoning_status = "missing"
            elif reasoning_tokens == 0:
                reasoning_status = "zero"
            else:
                reasoning_status = "positive"
        else:
            reasoning_status = "not_applicable"

        response_cost = _coerce_float(kwargs.get("response_cost"))
        if response_cost is None and event_type == "success":
            try:
                response_cost = completion_cost(
                    completion_response=response_obj,
                    model=model,
                    call_type=kwargs.get("call_type"),
                    custom_llm_provider=kwargs.get("custom_llm_provider"),
                    optional_params=kwargs.get("optional_params"),
                )
            except Exception:
                response_cost = None

        metadata = kwargs.get("metadata")
        if not isinstance(metadata, dict):
            metadata = None

        event = {
            "event_idx": None,
            "event_type": event_type,
            "timestamp": _utc_now_iso(),
            "run_id": self.run_id,
            "litellm_call_id": litellm_call_id,
            "model": model,
            "provider": kwargs.get("custom_llm_provider"),
            "call_type": kwargs.get("call_type"),
            "duration_ms": _extract_duration_ms(start_time, end_time),
            "response_cost_usd": response_cost,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "reasoning_tokens": reasoning_tokens,
                "raw": usage,
            },
            "reasoning_model": is_reasoning,
            "reasoning_token_status": reasoning_status,
            "error": _extract_error_text(response_obj) if event_type == "failure" else None,
            "metadata": _json_safe(metadata) if metadata else None,
        }

        with self._lock:
            if litellm_call_id is not None:
                event_key = (event_type, litellm_call_id)
                if event_key in self._seen_event_keys:
                    return
                self._seen_event_keys.add(event_key)

            self._event_count += 1
            event["event_idx"] = self._event_count
            with self.events_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_json_safe(event), sort_keys=True) + "\n")

            self._stats["events_logged"] += 1
            if event_type == "success":
                self._stats["success_calls"] += 1
            else:
                self._stats["failure_calls"] += 1

            if response_cost is not None:
                self._stats["cost_total_usd"] += response_cost
            else:
                self._stats["cost_missing_calls"] += 1

            if prompt_tokens is None and completion_tokens is None and total_tokens is None:
                self._stats["usage_missing_calls"] += 1

            if prompt_tokens is not None:
                self._stats["prompt_tokens_total"] += prompt_tokens
            if completion_tokens is not None:
                self._stats["completion_tokens_total"] += completion_tokens
            if total_tokens is not None:
                self._stats["total_tokens_total"] += total_tokens

            if is_reasoning:
                self._stats["reasoning_model_calls"] += 1
                if reasoning_tokens is None:
                    self._stats["reasoning_tokens_missing_calls"] += 1
                else:
                    self._stats["reasoning_tokens_reported_calls"] += 1
                    self._stats["reasoning_tokens_total"] += reasoning_tokens
                    if reasoning_tokens == 0:
                        self._stats["reasoning_tokens_zero_calls"] += 1
                    elif reasoning_tokens > 0:
                        self._stats["reasoning_tokens_positive_calls"] += 1

            per_model = self._stats["per_model"].setdefault(
                model,
                {
                    "success_calls": 0,
                    "failure_calls": 0,
                    "cost_total_usd": 0.0,
                    "cost_missing_calls": 0,
                    "usage_missing_calls": 0,
                    "prompt_tokens_total": 0,
                    "completion_tokens_total": 0,
                    "total_tokens_total": 0,
                    "reasoning_model_calls": 0,
                    "reasoning_tokens_total": 0,
                    "reasoning_tokens_reported_calls": 0,
                    "reasoning_tokens_missing_calls": 0,
                    "reasoning_tokens_zero_calls": 0,
                    "reasoning_tokens_positive_calls": 0,
                },
            )
            if event_type == "success":
                per_model["success_calls"] += 1
            else:
                per_model["failure_calls"] += 1
            if response_cost is not None:
                per_model["cost_total_usd"] += response_cost
            else:
                per_model["cost_missing_calls"] += 1
            if prompt_tokens is None and completion_tokens is None and total_tokens is None:
                per_model["usage_missing_calls"] += 1
            if prompt_tokens is not None:
                per_model["prompt_tokens_total"] += prompt_tokens
            if completion_tokens is not None:
                per_model["completion_tokens_total"] += completion_tokens
            if total_tokens is not None:
                per_model["total_tokens_total"] += total_tokens
            if is_reasoning:
                per_model["reasoning_model_calls"] += 1
                if reasoning_tokens is None:
                    per_model["reasoning_tokens_missing_calls"] += 1
                else:
                    per_model["reasoning_tokens_reported_calls"] += 1
                    per_model["reasoning_tokens_total"] += reasoning_tokens
                    if reasoning_tokens == 0:
                        per_model["reasoning_tokens_zero_calls"] += 1
                    elif reasoning_tokens > 0:
                        per_model["reasoning_tokens_positive_calls"] += 1

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record_event("success", kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record_event("success", kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._record_event("failure", kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._record_event("failure", kwargs, response_obj, start_time, end_time)

    def finalize(self) -> dict[str, Any]:
        with self._lock:
            if self._finalized and self._summary_cache is not None:
                return self._summary_cache

            summary = {
                **self._stats,
                "end_time": _utc_now_iso(),
            }
            self.summary_file.write_text(
                json.dumps(_json_safe(summary), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            self._summary_cache = summary
            self._finalized = True
            return summary


def install_spend_tracker(run_id: str, log_dir: Path) -> RunSpendTracker:
    tracker = RunSpendTracker(run_id=run_id, log_dir=log_dir)
    if tracker not in litellm.callbacks:
        litellm.callbacks.append(tracker)
    return tracker


def uninstall_spend_tracker(tracker: RunSpendTracker) -> None:
    callback_list_names = [
        "callbacks",
        "input_callback",
        "success_callback",
        "failure_callback",
        "_async_input_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ]
    for list_name in callback_list_names:
        callback_list = getattr(litellm, list_name, None)
        if not isinstance(callback_list, list):
            continue
        while tracker in callback_list:
            callback_list.remove(tracker)
