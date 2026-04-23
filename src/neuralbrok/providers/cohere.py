"""
Cohere provider adapter.

Translates between OpenAI wire format and Cohere's /v2/chat API.
Handles streaming (event-stream) and non-streaming responses.
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

_BASE_URL = "https://api.cohere.com/v2"


def _split_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Split OpenAI messages into (last_user_message, chat_history).

    Cohere expects the current user turn as 'message' and prior
    turns as 'chat_history' with USER/CHATBOT roles.
    """
    if not messages:
        return "", []

    # Find the last user message
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx == -1:
        return "", []

    current_message = messages[last_user_idx].get("content", "")
    history_msgs = messages[:last_user_idx]

    chat_history = []
    for msg in history_msgs:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            continue  # Skip system messages in history
        cohere_role = "CHATBOT" if role == "assistant" else "USER"
        chat_history.append({"role": cohere_role, "message": content})

    return current_message, chat_history


class CohereProvider(BaseProvider):
    """Adapter for Cohere AI chat API.

    Translates OpenAI chat/completions format to Cohere's /v2/chat
    with role mapping and streaming event translation.
    """

    SUPPORTED_MODELS = ["command-r-plus", "command-r", "command", "command-light"]

    def __init__(self, name: str, api_key: str, base_url: str = _BASE_URL):
        super().__init__(name=name, provider_type="cloud")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def chat(
        self, payload: dict, stream: bool = True
    ) -> AsyncIterator[str]:
        """Forward request to Cohere and yield OpenAI-compatible SSE."""
        url = f"{self.base_url}/chat"
        chunk_id = f"chatcmpl-nb-{self.name}-{int(time.time())}"
        model = payload.get("model", "command-r")

        message, chat_history = _split_messages(payload.get("messages", []))

        request_body = {
            "model": model,
            "message": message,
            "chat_history": chat_history,
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

                            if line.startswith("data: "):
                                raw = line[6:]
                                try:
                                    event = json.loads(raw)
                                except json.JSONDecodeError:
                                    continue

                                event_type = event.get("event_type", "")
                                if event_type == "text-generation":
                                    text = event.get("text", "")
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
                                elif event_type == "stream-end":
                                    yield "data: [DONE]\n\n"
                                    break

                        yield "data: [DONE]\n\n"

                else:
                    response = await client.post(
                        url, json=request_body, headers=headers
                    )
                    if response.status_code == 429:
                        raise RateLimitError(self.name)
                    response.raise_for_status()
                    data = response.json()

                    # Cohere v2 non-streaming: message.content[0].text
                    content_blocks = (
                        data.get("message", {}).get("content", [])
                    )
                    text = "".join(
                        block.get("text", "")
                        for block in content_blocks
                        if block.get("type") == "text"
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
        except Exception as e:
            raise ProviderError(self.name, str(e))

    async def health_check(self) -> bool:
        """Check Cohere API reachability."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    "https://api.cohere.com/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                return r.status_code == 200
        except Exception:
            return False
