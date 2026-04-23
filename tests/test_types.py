"""Tests for type definitions."""
import pytest
from datetime import datetime

from neuralbrok.types import (
    OpenAIRequest,
    OpenAIResponse,
    RoutingMetadata,
    VramSnapshot,
)


class TestOpenAIRequest:
    """Tests for OpenAIRequest model."""

    def test_valid_request(self):
        """Test creating a valid OpenAI request."""
        req = OpenAIRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert req.model == "gpt-4"
        assert len(req.messages) == 1
        assert req.temperature == 0.7
        assert req.max_tokens == 2048
        assert req.stream is False

    def test_request_with_stream(self):
        """Test request with streaming enabled."""
        req = OpenAIRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        assert req.stream is True

    def test_temperature_validation(self):
        """Test temperature bounds validation."""
        # Valid range: 0.0 to 2.0
        OpenAIRequest(
            model="gpt-4",
            messages=[],
            temperature=0.0,
        )
        OpenAIRequest(
            model="gpt-4",
            messages=[],
            temperature=2.0,
        )

        # Invalid: out of bounds
        with pytest.raises(ValueError):
            OpenAIRequest(
                model="gpt-4",
                messages=[],
                temperature=2.5,
            )

    def test_max_tokens_validation(self):
        """Test max_tokens bounds validation."""
        OpenAIRequest(
            model="gpt-4",
            messages=[],
            max_tokens=1,
        )

        with pytest.raises(ValueError):
            OpenAIRequest(
                model="gpt-4",
                messages=[],
                max_tokens=0,
            )


class TestOpenAIResponse:
    """Tests for OpenAIResponse model."""

    def test_valid_response(self):
        """Test creating a valid response."""
        resp = OpenAIResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="gpt-4",
            choices=[{"index": 0, "delta": {"content": "Hello"}}],
        )
        assert resp.id == "chatcmpl-123"
        assert resp.object == "chat.completion.chunk"


class TestRoutingMetadata:
    """Tests for RoutingMetadata model."""

    def test_valid_metadata(self):
        """Test creating valid routing metadata."""
        meta = RoutingMetadata(
            backend_chosen="ollama",
            vram_used_gb=2.5,
            vram_free_gb=5.5,
            latency_ms=10.5,
        )
        assert meta.backend_chosen == "ollama"
        assert meta.vram_used_gb == 2.5


class TestVramSnapshot:
    """Tests for VramSnapshot dataclass."""

    def test_valid_snapshot(self):
        """Test creating a valid VRAM snapshot."""
        now = datetime.now()
        snap = VramSnapshot(
            gpu_id=0,
            vram_used_gb=2.5,
            vram_free_gb=5.5,
            timestamp=now,
        )
        assert snap.gpu_id == 0
        assert snap.vram_used_gb == 2.5
        assert snap.vram_free_gb == 5.5
        assert snap.timestamp == now
