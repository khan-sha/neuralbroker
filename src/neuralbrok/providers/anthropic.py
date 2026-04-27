"""
Anthropic provider adapter.

Translates between OpenAI wire format and Anthropic's /v1/messages API.
Handles streaming (SSE content_block_delta events) and non-streaming.
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

_BASE_URL = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    """Adapter for Anthropic Claude API.

    Translates OpenAI chat/completions format to Anthropic /v1/messages
    and translates the response back to OpenAI format.
    """

    SUPPORTED_MODELS = [
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-haiku-3-5",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
    ]

    def __init__(self, name: str, api_key: str, auth_type: str = "api_key"):
        super().__init__(name=name, provider_type="cloud")
        self.api_key = api_key
        self.auth_type = auth_type  # "api_key" or "oauth_bearer"
        self.base_url = _BASE_URL

    def _auth_headers(self) -> dict:
        if self.auth_type == "oauth_bearer":
            return {
                "Authorization": f"Bearer {self.api_key}",
                "anthropic-beta": "oauth-2025-04-20",
                "anthropic-version": _ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            }
        return {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    async def chat(
        self, payload: dict, stream: bool = True
    ) -> AsyncIterator[str]:
        """Forward request to Anthropic and yield OpenAI-compatible SSE."""
        url = f"{self.base_url}/v1/messages"
        chunk_id = f"chatcmpl-nb-{self.name}-{int(time.time())}"
        model = payload.get("model", "claude-3-haiku-20240307")

        # Separate system prompt from messages if present
        messages = payload.get("messages", [])
        system_content = None
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                filtered_messages.append(msg)

        request_body: dict = {
            "model": model,
            "max_tokens": payload.get("max_tokens", 2048),
            "messages": filtered_messages,
            "stream": stream,
        }
        if system_content:
            request_body["system"] = system_content

        headers = self._auth_headers()

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

                            # Anthropic SSE: "event: ..." then "data: ..."
                            if line.startswith("event: message_stop"):
                                yield "data: [DONE]\n\n"
                                break

                            if line.startswith("data: "):
                                raw = line[6:]
                                try:
                                    event = json.loads(raw)
                                except json.JSONDecodeError:
                                    continue

                                # content_block_delta carries the text
                                if event.get("type") == "content_block_delta":
                                    text = (
                                        event.get("delta", {}).get("text", "")
                                    )
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

                                elif event.get("type") == "message_stop":
                                    yield "data: [DONE]\n\n"
                                    break
                else:
                    response = await client.post(
                        url, json=request_body, headers=headers
                    )
                    if response.status_code == 429:
                        raise RateLimitError(self.name)
                    response.raise_for_status()
                    data = response.json()

                    # Extract text from content blocks
                    content_blocks = data.get("content", [])
                    text = "".join(
                        block.get("text", "")
                        for block in content_blocks
                        if block.get("type") == "text"
                    )

                    usage = data.get("usage", {})
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
                        "usage": {
                            "prompt_tokens": usage.get("input_tokens", 0),
                            "completion_tokens": usage.get("output_tokens", 0),
                            "total_tokens": (
                                usage.get("input_tokens", 0)
                                + usage.get("output_tokens", 0)
                            ),
                        },
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

    async def health_check(self) -> bool:
        """Check Anthropic API reachability."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    f"{self.base_url}/v1/models",
                    headers=self._auth_headers(),
                )
                return r.status_code in (200, 404)  # 404 = reachable but endpoint varies
        except Exception:
            return False
