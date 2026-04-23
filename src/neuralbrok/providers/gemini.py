"""
Google Gemini provider adapter.

Translates between OpenAI wire format and Google's Generative Language API.
Handles streaming (SSE alt=sse) and non-streaming requests.
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

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def _to_gemini_messages(messages: list[dict]) -> list[dict]:
    """Translate OpenAI messages to Gemini contents format.

    Maps OpenAI 'assistant' role to Gemini 'model' role.
    Skips system messages (not supported in contents; handled separately).
    """
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            # Gemini doesn't have a system role in contents
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({
            "role": gemini_role,
            "parts": [{"text": content}],
        })
    return contents


class GeminiProvider(BaseProvider):
    """Adapter for Google Gemini API.

    Translates OpenAI chat/completions format to Gemini's
    generateContent / streamGenerateContent endpoints.
    """

    SUPPORTED_MODELS = [
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.0-pro",
    ]

    def __init__(self, name: str, api_key: str):
        super().__init__(name=name, provider_type="cloud")
        self.api_key = api_key

    async def chat(
        self, payload: dict, stream: bool = True
    ) -> AsyncIterator[str]:
        """Forward request to Gemini and yield OpenAI-compatible SSE."""
        model = payload.get("model", "gemini-1.5-flash")
        chunk_id = f"chatcmpl-nb-{self.name}-{int(time.time())}"
        contents = _to_gemini_messages(payload.get("messages", []))
        request_body = {"contents": contents}

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                if stream:
                    url = (
                        f"{_BASE}/{model}:streamGenerateContent"
                        f"?key={self.api_key}&alt=sse"
                    )
                    async with client.stream(
                        "POST", url, json=request_body
                    ) as response:
                        if response.status_code == 429:
                            raise RateLimitError(self.name)
                        response.raise_for_status()

                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith("data: "):
                                raw = line[6:].strip()
                                if raw == "[DONE]":
                                    yield "data: [DONE]\n\n"
                                    break
                                try:
                                    event = json.loads(raw)
                                except json.JSONDecodeError:
                                    continue

                                candidates = event.get("candidates", [])
                                if candidates:
                                    parts = (
                                        candidates[0]
                                        .get("content", {})
                                        .get("parts", [])
                                    )
                                    text = "".join(
                                        p.get("text", "") for p in parts
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

                        yield "data: [DONE]\n\n"

                else:
                    url = (
                        f"{_BASE}/{model}:generateContent"
                        f"?key={self.api_key}"
                    )
                    response = await client.post(url, json=request_body)
                    if response.status_code == 429:
                        raise RateLimitError(self.name)
                    response.raise_for_status()
                    data = response.json()

                    text = (
                        data["candidates"][0]["content"]["parts"][0]["text"]
                    )
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
        except (KeyError, IndexError) as e:
            raise ProviderError(self.name, f"Unexpected response shape: {e}")
        except Exception as e:
            raise ProviderError(self.name, str(e))

    async def health_check(self) -> bool:
        """Check Gemini API reachability."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    f"{_BASE}?key={self.api_key}"
                )
                return r.status_code == 200
        except Exception:
            return False
