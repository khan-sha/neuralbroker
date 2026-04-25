"""
Ollama provider adapter.

Translates between OpenAI wire format and Ollama's /api/chat format.
Handles streaming and non-streaming responses.
"""
import json
import time
from typing import AsyncIterator

import httpx

from neuralbrok.providers.base import (
    BaseProvider,
    ProviderError,
    OOMError,
    BackendUnavailableError,
)

# Patterns that indicate an OOM in Ollama's response
OOM_PATTERNS = [
    "CUDA out of memory",
    "out of memory",
    "OOM",
    "device-side assert",
    "cudaMalloc failed",
]


class OllamaProvider(BaseProvider):
    """Adapter for Ollama backend.

    Transforms OpenAI format → Ollama /api/chat and streams SSE back.
    """

    # Local server — model depends on what's pulled; empty means no filtering
    SUPPORTED_MODELS: list[str] = []

    def __init__(self, name: str, host: str):
        super().__init__(name=name, provider_type="local")
        self.base_url = f"http://{host}" if "://" not in host else host

    async def chat(
        self, payload: dict, stream: bool = True
    ) -> AsyncIterator[str]:
        """Forward request to Ollama and yield OpenAI-compatible SSE chunks."""
        # Transform OpenAI → Ollama format
        ollama_payload = {
            "model": payload.get("model", ""),
            "messages": payload.get("messages", []),
            "stream": stream,
            "options": {},
        }

        # Map OpenAI params to Ollama options
        if "temperature" in payload and payload["temperature"] is not None:
            ollama_payload["options"]["temperature"] = payload["temperature"]
        if "max_tokens" in payload and payload["max_tokens"] is not None:
            ollama_payload["options"]["num_predict"] = payload["max_tokens"]
        if "top_p" in payload and payload["top_p"] is not None:
            ollama_payload["options"]["top_p"] = payload["top_p"]
        if "stop" in payload and payload["stop"] is not None:
            ollama_payload["stop"] = payload["stop"]

        url = f"{self.base_url}/api/chat"
        chunk_id = f"chatcmpl-nb-{self.name}-{int(time.time())}"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                if stream:
                    async with client.stream(
                        "POST", url, json=ollama_payload
                    ) as response:
                        response.raise_for_status()

                        async for line in response.aiter_lines():
                            if not line.strip():
                                continue

                            try:
                                chunk = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            content = chunk.get("message", {}).get("content", "")
                            done = chunk.get("done", False)

                            # Skip empty non-terminal chunks (thinking phase, etc.)
                            if not content and not done:
                                continue

                            # Check for OOM in content
                            for pattern in OOM_PATTERNS:
                                if pattern.lower() in content.lower():
                                    raise OOMError(self.name, content)

                            openai_chunk = {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": payload.get("model", ""),
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content} if content else {},
                                        "finish_reason": "stop" if done else None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(openai_chunk)}\n\n"

                        yield "data: [DONE]\n\n"
                else:
                    # Non-streaming: collect full response
                    response = await client.post(url, json=ollama_payload)
                    response.raise_for_status()
                    data = response.json()

                    content = data.get("message", {}).get("content", "")

                    # Check for OOM
                    for pattern in OOM_PATTERNS:
                        if pattern.lower() in content.lower():
                            raise OOMError(self.name, content)

                    openai_response = {
                        "id": chunk_id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": payload.get("model", ""),
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": data.get("prompt_eval_count", 0),
                            "completion_tokens": data.get("eval_count", 0),
                            "total_tokens": (
                                data.get("prompt_eval_count", 0)
                                + data.get("eval_count", 0)
                            ),
                        },
                    }
                    yield json.dumps(openai_response)

        except (OOMError, ProviderError):
            raise
        except httpx.ConnectError as e:
            raise BackendUnavailableError(self.name, f"Connection refused: {e}")
        except httpx.TimeoutException as e:
            raise BackendUnavailableError(self.name, f"Timeout: {e}")
        except httpx.HTTPStatusError as e:
            raise ProviderError(self.name, f"HTTP {e.response.status_code}: {e}")
        except Exception as e:
            raise ProviderError(self.name, str(e))

    async def health_check(self) -> bool:
        """Check if Ollama is running by hitting /api/tags."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False
