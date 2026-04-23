"""
Backend adapters for forwarding and transforming requests.

DEPRECATED: Use src.providers instead. This module is kept for
backward compatibility with existing tests and imports.
"""
import json
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

import httpx

from neuralbrok.types import OpenAIRequest


class BaseBackendAdapter(ABC):
    """Abstract base for backend implementations.

    Deprecated: Use src.providers.BaseProvider instead.
    """

    @abstractmethod
    async def forward_request(
        self, request: OpenAIRequest
    ) -> AsyncIterator[str]:
        """Forward request to backend and yield SSE chunks.

        Yields:
            Strings in format "data: {...}\\n\\n" or sentinel "[DONE]".
        """
        pass


class OllamaBackend(BaseBackendAdapter):
    """Adapter for Ollama backend.

    Transforms OpenAI format to Ollama /api/chat format and streams SSE.
    """

    def __init__(self, host: str):
        """Initialize Ollama adapter.

        Args:
            host: Ollama host URL (e.g., "localhost:11434").
        """
        self.base_url = f"http://{host}" if "://" not in host else host

    async def forward_request(
        self, request: OpenAIRequest
    ) -> AsyncIterator[str]:
        """Forward request to Ollama and stream response as SSE.

        Args:
            request: OpenAI-formatted request.

        Yields:
            SSE-formatted chunk strings.

        Raises:
            httpx.HTTPError: On network or server error.
        """
        # Transform OpenAI request to Ollama format
        ollama_request = {
            "model": request.model,
            "messages": request.messages,
            "stream": True,
            "options": {
                "temperature": request.temperature or 0.7,
            },
        }

        url = f"{self.base_url}/api/chat"

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                url,
                json=ollama_request,
            ) as response:
                response.raise_for_status()

                # Stream response line by line
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        chunk = json.loads(line)
                        # Transform Ollama response to OpenAI SSE format
                        openai_chunk = {
                            "id": "chatcmpl-local",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": chunk.get("message", {}).get(
                                            "content", ""
                                        )
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(openai_chunk)}\n\n"
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue

                # Emit done sentinel
                yield "data: [DONE]\n\n"


class GroqBackend(BaseBackendAdapter):
    """Adapter for Groq backend.

    Groq API is OpenAI-compatible, so minimal transformation needed.
    """

    def __init__(self, base_url: str, api_key: str):
        """Initialize Groq adapter.

        Args:
            base_url: Groq API base URL.
            api_key: Groq API key.
        """
        self.base_url = base_url
        self.api_key = api_key

    async def forward_request(
        self, request: OpenAIRequest
    ) -> AsyncIterator[str]:
        """Forward request to Groq and stream response as SSE.

        Args:
            request: OpenAI-formatted request (used as-is).

        Yields:
            SSE-formatted chunk strings.

        Raises:
            httpx.HTTPError: On network or server error.
        """
        url = f"{self.base_url}/chat/completions"

        # Groq is OpenAI-compatible; send request as-is
        groq_request = request.model_dump(exclude_none=True)
        groq_request["stream"] = True

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                url,
                json=groq_request,
                headers=headers,
            ) as response:
                response.raise_for_status()

                # Stream SSE response line by line
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    # Groq returns OpenAI format natively
                    if line.startswith("data: "):
                        yield f"{line}\n"
                    elif line == "data: [DONE]":
                        yield "data: [DONE]\n\n"
