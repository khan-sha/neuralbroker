"""
Azure OpenAI provider adapter.

Azure uses a deployment-based URL and 'api-key' header instead of
'Authorization: Bearer'. Otherwise identical to OpenAI format.
"""
import json
from typing import AsyncIterator

import httpx

from neuralbrok.providers.base import (
    BaseProvider,
    ProviderError,
    RateLimitError,
    BackendUnavailableError,
)

_API_VERSION = "2024-02-01"


class AzureOpenAIProvider(BaseProvider):
    """Adapter for Azure OpenAI Service.

    Builds a full deployment URL at init time:
      https://{endpoint}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version=...

    Auth uses 'api-key' header instead of 'Authorization: Bearer'.
    SUPPORTED_MODELS is empty — models depend on which deployment is configured.
    """

    SUPPORTED_MODELS: list[str] = []  # Dynamic based on deployment

    def __init__(
        self,
        name: str,
        api_key: str,
        endpoint: str,
        deployment: str,
    ):
        super().__init__(name=name, provider_type="cloud")
        self.api_key = api_key
        # Build the full URL once
        self.url = (
            f"https://{endpoint}.openai.azure.com/openai/deployments"
            f"/{deployment}/chat/completions?api-version={_API_VERSION}"
        )

    async def chat(
        self, payload: dict, stream: bool = True
    ) -> AsyncIterator[str]:
        """Forward request to Azure OpenAI deployment."""
        request_body = {**payload, "stream": stream}

        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                if stream:
                    async with client.stream(
                        "POST", self.url, json=request_body, headers=headers
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
                        self.url, json=request_body, headers=headers
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
