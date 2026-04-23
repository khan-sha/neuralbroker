"""
Abstract base provider and error types.

Every backend adapter (local or cloud) implements BaseProvider.
"""
import json
from abc import ABC, abstractmethod
from typing import AsyncIterator

import httpx


class ProviderError(Exception):
    """Raised when a provider encounters an error.

    Used to trigger circuit breaker logic in the router.
    """

    def __init__(self, provider: str, message: str, retryable: bool = True):
        self.provider = provider
        self.retryable = retryable
        super().__init__(f"[{provider}] {message}")


class OOMError(ProviderError):
    """CUDA/GPU out-of-memory detected in provider response."""

    def __init__(self, provider: str, message: str = "Out of memory"):
        super().__init__(provider, message, retryable=True)


class RateLimitError(ProviderError):
    """HTTP 429 rate limit from cloud provider."""

    def __init__(self, provider: str, message: str = "Rate limited"):
        super().__init__(provider, message, retryable=True)


class BackendUnavailableError(ProviderError):
    """Backend is not reachable (connection refused, timeout, etc.)."""

    def __init__(self, provider: str, message: str = "Backend unavailable"):
        super().__init__(provider, message, retryable=True)


class BaseProvider(ABC):
    """Abstract base for all LLM backend providers.

    Each provider implements chat() which accepts an OpenAI-format payload
    and yields SSE chunks as strings.
    """

    name: str
    provider_type: str  # "local" or "cloud"

    def __init__(self, name: str, provider_type: str):
        self.name = name
        self.provider_type = provider_type

    @abstractmethod
    async def chat(
        self, payload: dict, stream: bool = True
    ) -> AsyncIterator[str]:
        """Forward a chat completion request and yield SSE chunks.

        Args:
            payload: OpenAI-format request body.
            stream: Whether to stream the response.

        Yields:
            Strings in SSE format: "data: {...}\\n\\n" or "data: [DONE]\\n\\n".

        Raises:
            ProviderError: On any backend error (connection, OOM, rate limit).
        """
        pass

    async def health_check(self) -> bool:
        """Check if the provider is reachable and healthy.

        Returns:
            True if healthy, False otherwise.
        """
        return True

    def _make_openai_error_response(self, status: int, message: str) -> str:
        """Create an OpenAI-format error response."""
        return json.dumps({
            "error": {
                "message": message,
                "type": "server_error",
                "code": status,
            }
        })
