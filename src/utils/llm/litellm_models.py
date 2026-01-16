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


@dataclass
class LiteLLMModel:
    """Lightweight model wrapper for LiteLLM.

    Attributes:
        id: Model identifier in LiteLLM format (e.g., "openai/gpt-4o")
    """

    id: str

    @property
    def provider_cls(self) -> str:
        """Infer provider class name from model ID prefix for rate limiting."""
        prefix = self.id.split("/")[0] if "/" in self.id else "openai"
        return PROVIDER_MAP.get(prefix, "OpenAIProvider")

    def get_response(
        self, prompt: str, temperature: float = 0.0, max_tokens: int = 50
    ) -> str:
        """Synchronous call using LiteLLM's completion.

        Args:
            prompt: The prompt text to send to the model
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            The model's response text
        """
        response = completion(
            model=self.id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    async def get_response_async(
        self, prompt: str, temperature: float = 0.0, max_tokens: int = 50
    ) -> str:
        """Native async call using LiteLLM's acompletion.

        Args:
            prompt: The prompt text to send to the model
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            The model's response text
        """
        response = await acompletion(
            model=self.id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
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
    secrets = {
        "OPENAI_API_KEY": "openai-api-key",
        "ANTHROPIC_API_KEY": "anthropic-api-key",
        "GOOGLE_API_KEY": "google-api-key",
        "TOGETHER_API_KEY": "together-api-key",
        "MISTRAL_API_KEY": "mistral-api-key",
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
