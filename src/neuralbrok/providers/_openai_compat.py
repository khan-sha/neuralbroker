"""
OpenAI-compatible cloud provider adapter.

Shared base for Groq, Together, OpenAI, and any other provider that speaks
the OpenAI /v1/chat/completions wire format natively.
"""
import json
import time
from typing import AsyncIterator

import httpx

from neuralbrok.providers.base import (
    BaseProvider,
    ProviderError,
    RateLimitError,
    BackendUnavailableError,
)


class OpenAICompatibleProvider(BaseProvider):
    """Base adapter for any OpenAI-compatible cloud API.

    Groq, Together, and OpenAI all speak the same wire format.
    Each subclass just sets name, base_url, and api_key.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        provider_type: str = "cloud",
    ):
        super().__init__(name=name, provider_type=provider_type)
        # Ensure base_url doesn't have trailing slash
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def chat(
        self, payload: dict, stream: bool = True
    ) -> AsyncIterator[str]:
        """Forward request to OpenAI-compatible endpoint."""
        url = f"{self.base_url}/chat/completions"

        # Clean payload — send as-is since the API is OpenAI-native
        request_body = {**payload, "stream": stream}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                if stream:
                    async with client.stream(
                        "POST", url, json=request_body, headers=headers
                    ) as response:
                        if response.status_code == 429:
                            raise RateLimitError(self.name)
                        response.raise_for_status()

                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line:
                                continue

                            if line == "data: [DONE]":
                                yield "data: [DONE]\n\n"
                                break

                            if line.startswith("data: "):
                                yield f"{line}\n\n"

                else:
                    # Non-streaming
                    response = await client.post(
                        url, json=request_body, headers=headers
                    )
                    if response.status_code == 429:
                        raise RateLimitError(self.name)
                    response.raise_for_status()
                    yield response.text

        except (ProviderError, RateLimitError):
            raise
        except httpx.ConnectError as e:
            raise BackendUnavailableError(self.name, f"Connection refused: {e}")
        except httpx.TimeoutException as e:
            raise BackendUnavailableError(self.name, f"Timeout: {e}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(self.name)
            raise ProviderError(
                self.name, f"HTTP {e.response.status_code}: {e}"
            )
        except Exception as e:
            raise ProviderError(self.name, str(e))

    async def health_check(self) -> bool:
        """Check provider reachability with a HEAD to /models."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                return r.status_code == 200
        except Exception:
            return False
