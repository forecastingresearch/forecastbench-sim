"""LiteLLM model wrapper for CivBench evaluation.

Simple wrapper around LiteLLM for unified LLM access.
API keys are read from environment variables:
- ANTHROPIC_API_KEY
- OPENAI_API_KEY
- GOOGLE_API_KEY / GEMINI_API_KEY
- TOGETHER_API_KEY
- MISTRAL_API_KEY
- XAI_API_KEY

Keys can be loaded from GCP Secret Manager using load_api_keys_from_gcp().
"""

import os
from dataclasses import dataclass

from litellm import acompletion, completion


def load_api_keys_from_gcp(project_id: str | None = None) -> None:
    """Load API keys from GCP Secret Manager into environment variables.

    Args:
        project_id: GCP project ID. If None, reads from GOOGLE_CLOUD_PROJECT env var.
    """
    from google.cloud import secretmanager

    project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("No GCP project ID. Set GOOGLE_CLOUD_PROJECT or pass project_id.")

    client = secretmanager.SecretManagerServiceClient()

    secrets = {
        "OPENAI_API_KEY": "API_KEY_OPENAI",
        "ANTHROPIC_API_KEY": "API_KEY_ANTHROPIC",
        "GOOGLE_API_KEY": "API_KEY_GEMINI",
        "TOGETHER_API_KEY": "API_KEY_TOGETHERAI",
        "MISTRAL_API_KEY": "API_KEY_MISTRAL",
        "XAI_API_KEY": "API_KEY_XAI",
    }

    for env_var, secret_name in secrets.items():
        if os.environ.get(env_var):  # Skip if already set
            continue
        try:
            name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            os.environ[env_var] = response.payload.data.decode("UTF-8")
        except Exception:
            pass  # Secret doesn't exist or no access


# Models that don't support temperature parameter (reasoning models)
MODELS_WITHOUT_TEMPERATURE = {
    "o1", "o1-mini", "o1-preview",
    "o3", "o3-mini", "o3-pro",
    "o4-mini",
}

# Provider mapping from model ID prefix to rate limiter class name
PROVIDER_MAP = {
    "openai": "OpenAIProvider",
    "anthropic": "AnthropicProvider",
    "google": "GoogleProvider",
    "gemini": "GoogleProvider",
    "together": "TogetherProvider",
    "mistral": "MistralProvider",
    "xai": "XAIProvider",
}

# Prefix normalization for LiteLLM API calls
LITELLM_PREFIX_MAP = {
    "google": "gemini",  # LiteLLM uses gemini/ for Google AI Studio API
}


def _get_base_model_name(model_id: str) -> str:
    """Extract base model name without provider prefix and date suffix."""
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id
    return model_name.rsplit("-202", 1)[0]


def _supports_temperature(model_id: str) -> bool:
    """Check if a model supports the temperature parameter."""
    return _get_base_model_name(model_id) not in MODELS_WITHOUT_TEMPERATURE


@dataclass
class LiteLLMModel:
    """Lightweight model wrapper for LiteLLM.

    Attributes:
        id: Model identifier in LiteLLM format (e.g., "openai/gpt-4o")
    """

    id: str

    @property
    def _litellm_model_id(self) -> str:
        """Get the model ID normalized for LiteLLM API calls."""
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
    def is_reasoning_model(self) -> bool:
        """Check if this is a reasoning model that uses extended thinking."""
        return _get_base_model_name(self.id) in MODELS_WITHOUT_TEMPERATURE

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
            return max(2000, base_tokens * 4)
        return base_tokens

    def get_response(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> str:
        """Synchronous call using LiteLLM's completion."""
        kwargs = {
            "model": self._litellm_model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if self.supports_temperature:
            kwargs["temperature"] = temperature

        response = completion(**kwargs)
        return response.choices[0].message.content

    async def get_response_async(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> str:
        """Async call using LiteLLM's acompletion."""
        kwargs = {
            "model": self._litellm_model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if self.supports_temperature:
            kwargs["temperature"] = temperature

        response = await acompletion(**kwargs)
        return response.choices[0].message.content


def get_models(model_ids: list[str]) -> list[LiteLLMModel]:
    """Create model objects from a list of model IDs."""
    return [LiteLLMModel(id=mid) for mid in model_ids]
