"""
Tests for provider adapters.
"""
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import httpx

from neuralbrok.providers.base import ProviderError, OOMError, RateLimitError, BackendUnavailableError
from neuralbrok.providers.ollama import OllamaProvider
from neuralbrok.providers.groq import GroqProvider
from neuralbrok.providers.together import TogetherProvider
from neuralbrok.providers.openai_provider import OpenAIProvider
from neuralbrok.providers.llama_cpp import LlamaCppProvider


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_payload(model="llama3.1:8b"):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
        "stream": True,
    }


def _build_stream_mock(lines_fn, status_code=200):
    """Build a properly nested httpx AsyncClient mock for streaming.

    Handles: async with httpx.AsyncClient() as client:
                 async with client.stream(...) as response:
    """
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.status_code = status_code
    mock_response.aiter_lines = lines_fn

    # Inner context manager: client.stream(...)
    stream_cm = AsyncMock()
    stream_cm.__aenter__.return_value = mock_response
    stream_cm.__aexit__.return_value = None

    # The client instance
    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=stream_cm)

    # Outer context manager: httpx.AsyncClient(...)
    client_cm = AsyncMock()
    client_cm.__aenter__.return_value = mock_client
    client_cm.__aexit__.return_value = None

    return client_cm, mock_client, mock_response


# ── OllamaProvider ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestOllamaProvider:

    async def test_streams_sse_chunks(self):
        provider = OllamaProvider(name="test-ollama", host="localhost:11434")

        async def mock_lines():
            yield json.dumps({"message": {"content": "Hi"}, "done": False})
            yield json.dumps({"message": {"content": "!"}, "done": True})

        client_cm, mock_client, _ = _build_stream_mock(mock_lines)

        with patch("neuralbrok.providers.ollama.httpx.AsyncClient", return_value=client_cm):
            chunks = []
            async for chunk in provider.chat(make_payload(), stream=True):
                chunks.append(chunk)

            assert len(chunks) >= 2
            assert chunks[-1] == "data: [DONE]\n\n"
            assert all("data: " in c for c in chunks[:-1])

    async def test_detects_oom_in_stream(self):
        provider = OllamaProvider(name="test-ollama", host="localhost:11434")

        async def mock_lines():
            yield json.dumps({
                "message": {"content": "CUDA out of memory. Tried to allocate 2.34 GiB."},
                "done": False,
            })

        client_cm, _, _ = _build_stream_mock(mock_lines)

        with patch("neuralbrok.providers.ollama.httpx.AsyncClient", return_value=client_cm):
            with pytest.raises(OOMError):
                async for _ in provider.chat(make_payload(), stream=True):
                    pass

    async def test_connection_error_raises_backend_unavailable(self):
        provider = OllamaProvider(name="test-ollama", host="localhost:99999")

        # Mock client whose stream() raises ConnectError
        mock_client = MagicMock()
        mock_client.stream = MagicMock(side_effect=httpx.ConnectError("refused"))

        client_cm = AsyncMock()
        client_cm.__aenter__.return_value = mock_client
        client_cm.__aexit__.return_value = None

        with patch("neuralbrok.providers.ollama.httpx.AsyncClient", return_value=client_cm):
            with pytest.raises(BackendUnavailableError):
                async for _ in provider.chat(make_payload(), stream=True):
                    pass


# ── GroqProvider ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestGroqProvider:

    async def test_streams_openai_format(self):
        provider = GroqProvider(
            name="groq", base_url="https://api.groq.com/openai/v1", api_key="test"
        )

        async def mock_lines():
            yield "data: " + json.dumps({
                "id": "chatcmpl-123",
                "choices": [{"delta": {"content": "Hi"}}],
            })
            yield "data: [DONE]"

        client_cm, _, _ = _build_stream_mock(mock_lines)

        with patch("neuralbrok.providers._openai_compat.httpx.AsyncClient", return_value=client_cm):
            chunks = []
            async for chunk in provider.chat(make_payload(), stream=True):
                chunks.append(chunk)

            assert len(chunks) >= 1
            assert any("[DONE]" in c for c in chunks)

    async def test_auth_header_set(self):
        provider = GroqProvider(
            name="groq", base_url="https://api.groq.com/openai/v1", api_key="my-key"
        )

        async def mock_lines():
            yield "data: [DONE]"

        client_cm, mock_client, _ = _build_stream_mock(mock_lines)

        with patch("neuralbrok.providers._openai_compat.httpx.AsyncClient", return_value=client_cm):
            async for _ in provider.chat(make_payload(), stream=True):
                pass

            # Verify stream was called with auth header
            call_kwargs = mock_client.stream.call_args
            headers = call_kwargs.kwargs.get("headers", {})
            assert headers["Authorization"] == "Bearer my-key"


# ── TogetherProvider ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestTogetherProvider:

    async def test_provider_type_is_cloud(self):
        provider = TogetherProvider(
            name="together", base_url="https://api.together.xyz/v1", api_key="key"
        )
        assert provider.provider_type == "cloud"


# ── OpenAIProvider ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestOpenAIProvider:

    async def test_provider_type_is_cloud(self):
        provider = OpenAIProvider(
            name="openai", base_url="https://api.openai.com/v1", api_key="key"
        )
        assert provider.provider_type == "cloud"


# ── LlamaCppProvider ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestLlamaCppProvider:

    async def test_provider_type_is_local(self):
        provider = LlamaCppProvider(name="llama-cpp", host="localhost:8080")
        assert provider.provider_type == "local"

    async def test_base_url_includes_v1(self):
        provider = LlamaCppProvider(name="llama-cpp", host="localhost:8080")
        assert provider.base_url.endswith("/v1")


# ── Error Hierarchy ───────────────────────────────────────────────────────────

class TestErrorTypes:

    def test_provider_error_retryable_flag(self):
        err = ProviderError("test", "fail", retryable=True)
        assert err.retryable is True
        assert err.provider == "test"

    def test_oom_error_is_retryable(self):
        err = OOMError("ollama")
        assert err.retryable is True

    def test_rate_limit_error_is_retryable(self):
        err = RateLimitError("groq")
        assert err.retryable is True

    def test_backend_unavailable_is_retryable(self):
        err = BackendUnavailableError("ollama", "connection refused")
        assert err.retryable is True
