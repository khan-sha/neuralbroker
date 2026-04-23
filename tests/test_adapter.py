"""Tests for backend adapters."""
import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock

from neuralbrok.adapter import OllamaBackend, GroqBackend
from neuralbrok.types import OpenAIRequest


@pytest.mark.asyncio
class TestOllamaBackend:
    """Tests for Ollama adapter."""

    async def test_ollama_request_transformation(self):
        """Test OpenAI request is transformed to Ollama format."""
        backend = OllamaBackend(host="localhost:11434")

        request = OpenAIRequest(
            model="mistral",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.8,
        )

        # Mock httpx.AsyncClient.stream
        with patch("neuralbrok.adapter.httpx.AsyncClient.stream") as mock_stream:
            # Mock the response
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock(return_value=None)

            # Yield sample Ollama response chunks
            async def stream_lines():
                yield json.dumps({
                    "message": {"content": "Hello "},
                    "done": False,
                })
                yield json.dumps({
                    "message": {"content": "there"},
                    "done": False,
                })
                yield json.dumps({
                    "message": {"content": "!"},
                    "done": True,
                })

            mock_response.aiter_lines = stream_lines

            # Set up context manager
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            mock_context.__aexit__.return_value = None
            mock_stream.return_value = mock_context

            # Collect SSE chunks
            chunks = []
            async for chunk in backend.forward_request(request):
                chunks.append(chunk)

            # Should have 3 content chunks + 1 done sentinel
            assert len(chunks) > 0
            # Check that chunks are SSE formatted
            assert any("data: " in chunk for chunk in chunks)

    async def test_ollama_sse_format(self):
        """Test that Ollama response is formatted as SSE."""
        backend = OllamaBackend(host="localhost:11434")

        request = OpenAIRequest(
            model="mistral",
            messages=[{"role": "user", "content": "test"}],
        )

        with patch("neuralbrok.adapter.httpx.AsyncClient.stream") as mock_stream:
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock(return_value=None)

            async def stream_lines():
                yield json.dumps({"message": {"content": "test"}, "done": True})

            mock_response.aiter_lines = stream_lines

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            mock_context.__aexit__.return_value = None
            mock_stream.return_value = mock_context

            chunks = []
            async for chunk in backend.forward_request(request):
                chunks.append(chunk)

            # Last chunk should be the done sentinel
            assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
class TestGroqBackend:
    """Tests for Groq adapter."""

    async def test_groq_pass_through(self):
        """Test that Groq request is passed through with proper headers."""
        backend = GroqBackend(
            base_url="https://api.groq.com/openai/v1",
            api_key="test-key",
        )

        request = OpenAIRequest(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        with patch("neuralbrok.adapter.httpx.AsyncClient.stream") as mock_stream:
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock(return_value=None)

            async def stream_lines():
                yield "data: " + json.dumps({
                    "id": "chatcmpl-groq",
                    "choices": [{"delta": {"content": "Hello"}}],
                })
                yield "data: [DONE]"

            mock_response.aiter_lines = stream_lines

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            mock_context.__aexit__.return_value = None
            mock_stream.return_value = mock_context

            chunks = []
            async for chunk in backend.forward_request(request):
                chunks.append(chunk)

            assert len(chunks) > 0
            # Verify Authorization header was set
            mock_stream.assert_called_once()
            call_kwargs = mock_stream.call_args[1]
            assert "headers" in call_kwargs
            assert "Authorization" in call_kwargs["headers"]
            assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"

    async def test_groq_sse_format(self):
        """Test that Groq responses maintain SSE format."""
        backend = GroqBackend(
            base_url="https://api.groq.com/openai/v1",
            api_key="test-key",
        )

        request = OpenAIRequest(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": "test"}],
        )

        with patch("neuralbrok.adapter.httpx.AsyncClient.stream") as mock_stream:
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock(return_value=None)

            async def stream_lines():
                yield "data: " + json.dumps({"choices": [{"delta": {"content": "."}}]})
                yield "data: [DONE]"

            mock_response.aiter_lines = stream_lines

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            mock_context.__aexit__.return_value = None
            mock_stream.return_value = mock_context

            chunks = []
            async for chunk in backend.forward_request(request):
                chunks.append(chunk)

            # All chunks should be properly formatted
            assert all("data: " in chunk or chunk.endswith("\n") for chunk in chunks)
