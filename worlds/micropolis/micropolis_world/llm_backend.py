"""The one import boundary between micropolis and its LLM client.

Everything that talks to a model imports `acompletion` / `completion` /
`completion_cost` and the transient-error classes from here, so switching
backends is a matter of flipping which block below is live.

The two backends agree on: the call shape (`model`, `messages`, `timeout`),
`completion_cost(completion_response=...)`, the five exception classes the
retry loop in prompting.py catches, and `to_model_id()`, slug -> model id.

Callers hand `model` the slug (see model_ids.py: an OpenRouter id with an
optional `:suffix` picking a ModelSpec) and the backend translates internally;
`to_model_id` is exported for code that needs the underlying id, such as the
external-scores join.
"""

# Env vars the live backend needs beyond what GCP Secret Manager fills in.
# Checked once by module_globals.ensure_api_keys().
REQUIRED_ENV_KEYS = ("OPENROUTER_API_KEY",)

# --- OpenRouter (live) -----------------------------------------------------
from .openrouter_completion import (
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    acompletion,
    completion,
    completion_cost,
    to_model_id,
)

# The re-exports themselves: this module is a pass-through, so the names it
# imports are its public surface rather than unused imports. Both backend
# blocks above provide exactly these, which is what makes the swap a swap.
__all__ = (
    "APIConnectionError",
    "AuthenticationError",
    "InternalServerError",
    "RateLimitError",
    "ServiceUnavailableError",
    "Timeout",
    "acompletion",
    "completion",
    "completion_cost",
    "to_model_id",
)
