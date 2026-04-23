"""
OpenRouter provider adapter.

Subclasses OpenAICompatibleProvider but overrides chat() to inject
required HTTP-Referer and X-Title headers per OpenRouter docs.
"""
import json
from typing import AsyncIterator

import httpx

from neuralbrok.providers._openai_compat import OpenAICompatibleProvider
from neuralbrok.providers.base import (
    ProviderError,
    RateLimitError,
    BackendUnavailableError,
)

_EXTRA_HEADERS = {
    "HTTP-Referer": "https://neuralbrok.com",
    "X-Title": "NeuralBroker",
}


class OpenRouterProvider(OpenAICompatibleProvider):
    """Adapter for OpenRouter aggregation API.

    Last-resort fallback covering 100+ models when no direct provider
    has the requested model. Adds required HTTP-Referer and X-Title
    headers per OpenRouter's attribution requirements.
    """

    SUPPORTED_MODELS = [
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
        "meta-llama/llama-3.1-8b-instruct",
        "google/gemini-flash-1.5",
    ]

    async def chat(
        self, payload: dict, stream: bool = True
    ) -> AsyncIterator[str]:
        """Forward request to OpenRouter with extra identification headers."""
        url = f"{self.base_url}/chat/completions"
        request_body = {**payload, "stream": stream}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **_EXTRA_HEADERS,
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
