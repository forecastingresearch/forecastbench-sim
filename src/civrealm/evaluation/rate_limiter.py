"""Per-provider rate limiting for parallel LLM evaluation."""

import asyncio
from typing import Type


class ProviderRateLimiter:
    """
    Manages per-provider concurrency limits using asyncio semaphores.

    Different API providers have different rate limits. This class ensures
    we don't exceed those limits by using semaphores to control concurrent
    requests per provider.

    Default limits are conservative and can be adjusted based on your API tier:
    - OpenAI: 10 concurrent (Tier 1 allows 500 RPM)
    - Anthropic: 5 concurrent (default tier is 50 RPM)
    - Google: 10 concurrent (Gemini has generous limits)
    - Together: 10 concurrent (generally generous)
    - Mistral: 5 concurrent (varies by plan)

    Example:
        >>> rate_limiter = ProviderRateLimiter()
        >>> semaphore = rate_limiter.get_semaphore("OpenAIProvider")
        >>> async with semaphore:
        ...     response = await query_api(...)
    """

    # Conservative limits to avoid rate limiting
    # With multiple models per provider, these limits apply across all models
    # e.g., 5 Anthropic models with limit 2 = max 2 concurrent Anthropic calls
    DEFAULT_LIMITS = {
        "OpenAIProvider": 3,      # 5 OpenAI models, stagger them
        "AnthropicProvider": 2,   # 5 Anthropic models, stagger them
        "GoogleProvider": 3,      # 3 Google models
        "TogetherProvider": 3,    # 4 Together models
        "MistralProvider": 2,     # 1 Mistral model
        "XAIProvider": 2,         # No xAI models currently
    }

    def __init__(self, custom_limits: dict[str, int] | None = None):
        """
        Initialize rate limiter with optional custom limits.

        Args:
            custom_limits: Dict mapping provider class name to concurrency limit.
                          Overrides defaults for specified providers.
        """
        limits = {**self.DEFAULT_LIMITS, **(custom_limits or {})}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._limits = limits

    def get_semaphore(self, provider_cls: Type | str) -> asyncio.Semaphore:
        """
        Get the semaphore for a provider class.

        Semaphores are created lazily on first access.

        Args:
            provider_cls: Provider class or class name string

        Returns:
            asyncio.Semaphore for rate limiting this provider
        """
        # Handle both class and string
        if isinstance(provider_cls, str):
            name = provider_cls
        else:
            name = provider_cls.__name__

        # Create semaphore lazily (must be in async context)
        if name not in self._semaphores:
            limit = self._limits.get(name, 5)  # Default to 5 if unknown
            self._semaphores[name] = asyncio.Semaphore(limit)

        return self._semaphores[name]

    def get_limit(self, provider_cls: Type | str) -> int:
        """Get the concurrency limit for a provider."""
        if isinstance(provider_cls, str):
            name = provider_cls
        else:
            name = provider_cls.__name__
        return self._limits.get(name, 5)

    def __repr__(self) -> str:
        return f"ProviderRateLimiter(limits={self._limits})"
