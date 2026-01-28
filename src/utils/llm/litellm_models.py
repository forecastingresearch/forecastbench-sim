"""LiteLLM-based model wrapper for unified LLM access.

This module provides a lightweight wrapper around LiteLLM for calling
various LLM providers with a unified interface.

Usage:
    from utils.llm.litellm_models import get_models, configure_api_keys

    configure_api_keys(from_gcp=True)
    models = get_models(["openai/gpt-4o", "anthropic/claude-sonnet-4-5-20250929"])

    for model in models:
        response = await model.get_response_async("Hello!")
"""

import os
from dataclasses import dataclass

from litellm import acompletion, completion


# Models that don't support temperature parameter (reasoning models)
MODELS_WITHOUT_TEMPERATURE = {
    # O-series reasoning models
    "o1", "o1-mini", "o1-preview",
    "o3", "o3-mini", "o3-pro",
    "o4-mini",
    # GPT-5 series reasoning models
    "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro",
    "gpt-5-codex",
    "gpt-5.1", "gpt-5.1-chat", "gpt-5.1-codex", "gpt-5.1-codex-mini", "gpt-5.1-codex-max",
    "gpt-5.2", "gpt-5.2-pro", "gpt-5.2-codex",
}

# O-series models: support reasoning_effort low/medium/high
O_SERIES_MODELS = {
    "o1", "o1-mini", "o1-preview",
    "o3", "o3-mini", "o3-pro",
    "o4-mini",
}

# GPT-5 base models: support reasoning_effort minimal/low/medium/high
GPT5_BASE_MODELS = {
    "gpt-5", "gpt-5-mini", "gpt-5-nano",
    "gpt-5-codex",
}

# GPT-5-pro: only supports reasoning_effort=high (fixed)
GPT5_PRO_MODELS = {"gpt-5-pro"}

# GPT-5.1 models: support none/low/medium/high (default: none)
GPT51_MODELS = {
    "gpt-5.1", "gpt-5.1-chat", "gpt-5.1-codex", "gpt-5.1-codex-mini",
}

# GPT-5.1-codex-max: supports low/medium/high/xhigh
GPT51_CODEX_MAX_MODELS = {"gpt-5.1-codex-max"}

# GPT-5.2 models: support none/low/medium/high/xhigh (default: none)
GPT52_MODELS = {
    "gpt-5.2", "gpt-5.2-pro", "gpt-5.2-codex",
}

# All models that support reasoning_effort parameter
REASONING_EFFORT_MODELS = (
    O_SERIES_MODELS | GPT5_BASE_MODELS | GPT5_PRO_MODELS |
    GPT51_MODELS | GPT51_CODEX_MAX_MODELS | GPT52_MODELS
)


def _get_base_model_name(model_id: str) -> str:
    """Extract base model name without provider prefix and date suffix."""
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id
    # Remove date suffix like -2025-04-16 or -20251001
    return model_name.rsplit("-202", 1)[0]


def _supports_temperature(model_id: str) -> bool:
    """Check if a model supports the temperature parameter."""
    return _get_base_model_name(model_id) not in MODELS_WITHOUT_TEMPERATURE


def _get_reasoning_effort(model_id: str, requested_effort: str = "medium") -> str | None:
    """Get the appropriate reasoning_effort value for a model.

    Args:
        model_id: Model identifier
        requested_effort: Desired effort level (default: medium)

    Returns:
        The effort level to use, or None if model doesn't support reasoning_effort
    """
    base_name = _get_base_model_name(model_id)

    if base_name not in REASONING_EFFORT_MODELS:
        return None

    # gpt-5-pro only supports high
    if base_name in GPT5_PRO_MODELS:
        return "high"

    # All other reasoning models support medium
    return requested_effort


# Provider mapping from model ID prefix to rate limiter class name
PROVIDER_MAP = {
    "openai": "OpenAIProvider",
    "anthropic": "AnthropicProvider",
    "google": "GoogleProvider",
    "gemini": "GoogleProvider",  # Alias
    "together": "TogetherProvider",
    "mistral": "MistralProvider",
    "xai": "XAIProvider",
}

# Prefix normalization for LiteLLM API calls
# LiteLLM expects specific prefixes that may differ from user-friendly names
LITELLM_PREFIX_MAP = {
    "google": "gemini",  # LiteLLM uses gemini/ for Google AI Studio API
}


@dataclass
class LiteLLMModel:
    """Lightweight model wrapper for LiteLLM.

    Attributes:
        id: Model identifier in LiteLLM format (e.g., "openai/gpt-4o")
    """

    id: str

    @property
    def _litellm_model_id(self) -> str:
        """Get the model ID normalized for LiteLLM API calls.

        LiteLLM expects specific prefixes (e.g., 'gemini/' not 'google/').
        """
        if "/" in self.id:
            prefix, model_name = self.id.split("/", 1)
            normalized_prefix = LITELLM_PREFIX_MAP.get(prefix, prefix)
            return f"{normalized_prefix}/{model_name}"
        return self.id

    @property
    def provider_cls(self) -> str:
        """Infer provider class name from model ID prefix for rate limiting."""
        prefix = self.id.split("/")[0] if "/" in self.id else "openai"
        return PROVIDER_MAP.get(prefix, "OpenAIProvider")

    @property
    def supports_temperature(self) -> bool:
        """Check if this model supports the temperature parameter."""
        return _supports_temperature(self.id)

    @property
    def reasoning_effort(self) -> str | None:
        """Get the reasoning_effort setting for this model, or None if not supported."""
        return _get_reasoning_effort(self.id)

    @property
    def is_reasoning_model(self) -> bool:
        """Check if this is a reasoning model that uses extended thinking."""
        return _get_base_model_name(self.id) in REASONING_EFFORT_MODELS

    def effective_max_tokens(self, base_tokens: int) -> int:
        """Get effective max_tokens, accounting for reasoning models' token needs.

        Reasoning models consume tokens for both reasoning and output.
        This returns an appropriate token budget based on model type.

        Args:
            base_tokens: Desired output tokens for non-reasoning models

        Returns:
            Adjusted max_tokens value (higher for reasoning models)
        """
        if self.is_reasoning_model:
            # Reasoning models need more tokens since reasoning consumes budget
            # Use 4x multiplier with minimum of 2000 tokens
            return max(2000, base_tokens * 4)
        return base_tokens

    def get_response(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
        reasoning_effort: str = "medium",
    ) -> str:
        """Synchronous call using LiteLLM's completion.

        Args:
            prompt: The prompt text to send to the model
            temperature: Sampling temperature (0.0 = deterministic), ignored for reasoning models
            max_tokens: Maximum completion tokens (includes reasoning + output for reasoning models)
            reasoning_effort: Effort level for reasoning models (low/medium/high)

        Returns:
            The model's response text
        """
        kwargs = {
            "model": self._litellm_model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        # Only pass temperature for models that support it
        if self.supports_temperature:
            kwargs["temperature"] = temperature

        # Add reasoning_effort for reasoning models
        effort = _get_reasoning_effort(self.id, reasoning_effort)
        if effort:
            kwargs["reasoning_effort"] = effort

        response = completion(**kwargs)
        return response.choices[0].message.content

    async def get_response_async(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
        reasoning_effort: str = "medium",
    ) -> str:
        """Native async call using LiteLLM's acompletion.

        Args:
            prompt: The prompt text to send to the model
            temperature: Sampling temperature (0.0 = deterministic), ignored for reasoning models
            max_tokens: Maximum completion tokens (includes reasoning + output for reasoning models)
            reasoning_effort: Effort level for reasoning models (low/medium/high)

        Returns:
            The model's response text
        """
        kwargs = {
            "model": self._litellm_model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        # Only pass temperature for models that support it
        if self.supports_temperature:
            kwargs["temperature"] = temperature

        # Add reasoning_effort for reasoning models
        effort = _get_reasoning_effort(self.id, reasoning_effort)
        if effort:
            kwargs["reasoning_effort"] = effort

        response = await acompletion(**kwargs)
        return response.choices[0].message.content


def get_models(model_ids: list[str]) -> list[LiteLLMModel]:
    """Create model objects from a list of model IDs.

    Args:
        model_ids: List of model IDs in LiteLLM format (e.g., ["openai/gpt-4o"])

    Returns:
        List of LiteLLMModel objects
    """
    return [LiteLLMModel(id=mid) for mid in model_ids]


def configure_api_keys(from_gcp: bool = False, **kwargs) -> None:
    """Configure API keys from environment variables or GCP Secret Manager.

    LiteLLM reads API keys from environment variables automatically:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - GOOGLE_API_KEY (or GEMINI_API_KEY)
    - TOGETHER_API_KEY
    - MISTRAL_API_KEY

    Args:
        from_gcp: If True, fetch missing keys from GCP Secret Manager.
                  Requires GCP_PROJECT_ID environment variable.
        **kwargs: Explicit API keys to set (e.g., openai="sk-...", anthropic="...")
    """
    if from_gcp:
        _load_keys_from_gcp()

    # Override with explicit kwargs
    for key, value in kwargs.items():
        if value:
            env_var = f"{key.upper()}_API_KEY"
            os.environ[env_var] = value


def _load_keys_from_gcp() -> None:
    """Load API keys from GCP Secret Manager."""
    try:
        from google.cloud import secretmanager
    except ImportError:
        return  # GCP SDK not installed, skip

    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        return  # No project ID configured

    client = secretmanager.SecretManagerServiceClient()

    # Mapping of environment variable -> GCP secret name
    # Secret names match user's GCP Secret Manager naming convention (API_KEY_*)
    secrets = {
        "OPENAI_API_KEY": "API_KEY_OPENAI",
        "ANTHROPIC_API_KEY": "API_KEY_ANTHROPIC",
        "GOOGLE_API_KEY": "API_KEY_GEMINI",  # Use Gemini key for Google API
        "GEMINI_API_KEY": "API_KEY_GEMINI",
        "TOGETHER_API_KEY": "API_KEY_TOGETHERAI",
        "MISTRAL_API_KEY": "API_KEY_MISTRAL",
        "XAI_API_KEY": "API_KEY_XAI",
    }

    for env_var, secret_name in secrets.items():
        # Skip if already set
        if os.getenv(env_var):
            continue

        try:
            name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            os.environ[env_var] = response.payload.data.decode("UTF-8")
        except Exception:
            pass  # Secret doesn't exist or access denied, skip
