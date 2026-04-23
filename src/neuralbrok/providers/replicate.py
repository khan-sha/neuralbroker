"""
Replicate provider adapter.

Translates between OpenAI wire format and Replicate's polling-based
predictions API. Supports streaming via Replicate's stream URL when
available, with polling fallback.
"""
import asyncio
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

_BASE_URL = "https://api.replicate.com"
_POLL_INTERVAL = 0.5  # seconds


def _assemble_prompt(messages: list[dict]) -> str:
    """Assemble OpenAI messages into a single prompt string."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


class ReplicateProvider(BaseProvider):
    """Adapter for Replicate inference API.

    Uses Replicate's polling-based predictions API. Not OpenAI-compatible
    natively — assembles a prompt string from messages and polls for completion.
    For streaming, uses Replicate's stream URL if provided in the prediction
    response, otherwise falls back to polling.
    """

    SUPPORTED_MODELS = [
        "meta/llama-2-70b-chat",
        "mistralai/mixtral-8x7b-instruct-v0.1",
    ]

    def __init__(self, name: str, api_key: str, base_url: str = _BASE_URL):
        super().__init__(name=name, provider_type="cloud")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _auth_headers(self) -> dict:
        return {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat(
        self, payload: dict, stream: bool = True
    ) -> AsyncIterator[str]:
        """Submit prediction to Replicate and yield OpenAI-compatible response."""
        model = payload.get("model", "meta/llama-2-70b-chat")
        chunk_id = f"chatcmpl-nb-{self.name}-{int(time.time())}"
        prompt = _assemble_prompt(payload.get("messages", []))

        # Create prediction
        create_url = f"{self.base_url}/v1/models/{model}/predictions"
        create_body = {
            "input": {"prompt": prompt},
            "stream": stream,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                create_resp = await client.post(
                    create_url,
                    json=create_body,
                    headers=self._auth_headers(),
                )
                if create_resp.status_code == 429:
                    raise RateLimitError(self.name)
                create_resp.raise_for_status()
                prediction = create_resp.json()
                prediction_id = prediction.get("id")

                if not prediction_id:
                    raise ProviderError(
                        self.name, "No prediction ID in Replicate response"
                    )

                # Attempt streaming via stream URL if available
                stream_url = prediction.get("urls", {}).get("stream")
                if stream and stream_url:
                    async with client.stream(
                        "GET", stream_url, headers=self._auth_headers()
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith("data: "):
                                raw = line[6:]
                                if raw == "[DONE]":
                                    yield "data: [DONE]\n\n"
                                    break
                                try:
                                    event = json.loads(raw)
                                    text = event.get("output", "")
                                    if isinstance(text, list):
                                        text = "".join(text)
                                except (json.JSONDecodeError, TypeError):
                                    text = raw

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
                    return

                # Polling fallback (streaming or non-streaming)
                poll_url = (
                    prediction.get("urls", {}).get("get")
                    or f"{self.base_url}/v1/predictions/{prediction_id}"
                )
                while True:
                    await asyncio.sleep(_POLL_INTERVAL)
                    poll_resp = await client.get(
                        poll_url, headers=self._auth_headers()
                    )
                    poll_resp.raise_for_status()
                    result = poll_resp.json()
                    status = result.get("status")

                    if status == "failed":
                        error = result.get("error", "Unknown error")
                        raise ProviderError(
                            self.name, f"Prediction failed: {error}"
                        )

                    if status == "succeeded":
                        output = result.get("output", [])
                        if isinstance(output, list):
                            text = "".join(str(t) for t in output)
                        else:
                            text = str(output)

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
                        return

                    # Still processing — continue polling

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
