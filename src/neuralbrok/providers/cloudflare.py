"""
Cloudflare Workers AI provider adapter.

Edge inference via Cloudflare's per-model AI run endpoint.
Lowest latency for simple tasks due to edge deployment.
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


class CloudflareProvider(BaseProvider):
    """Adapter for Cloudflare Workers AI.

    Edge inference, lowest latency for simple tasks.
    Uses a per-model URL pattern:
      https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}
    """

    SUPPORTED_MODELS = [
        "@cf/meta/llama-3.1-8b-instruct",
        "@cf/mistral/mistral-7b-instruct-v0.1",
        "@cf/google/gemma-7b-it",
    ]

    def __init__(self, name: str, api_key: str, account_id: str):
        super().__init__(name=name, provider_type="cloud")
        self.api_key = api_key
        self.account_id = account_id

    def _model_url(self, model: str) -> str:
        return (
            f"https://api.cloudflare.com/client/v4/accounts"
            f"/{self.account_id}/ai/run/{model}"
        )

    async def chat(
        self, payload: dict, stream: bool = True
    ) -> AsyncIterator[str]:
        """Forward request to Cloudflare Workers AI."""
        model = payload.get("model", "@cf/meta/llama-3.1-8b-instruct")
        url = self._model_url(model)
        chunk_id = f"chatcmpl-nb-{self.name}-{int(time.time())}"

        request_body = {
            "messages": payload.get("messages", []),
            "stream": stream,
        }

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
                                raw = line[6:]
                                try:
                                    event = json.loads(raw)
                                except json.JSONDecodeError:
                                    continue

                                text = event.get("response", "")
                                openai_chunk = {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": text},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(openai_chunk)}\n\n"

                        yield "data: [DONE]\n\n"

                else:
                    response = await client.post(
                        url, json=request_body, headers=headers
                    )
                    if response.status_code == 429:
                        raise RateLimitError(self.name)
                    response.raise_for_status()
                    data = response.json()

                    text = data.get("result", {}).get("response", "")
                    openai_response = {
                        "id": chunk_id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": text,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield json.dumps(openai_response)

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
