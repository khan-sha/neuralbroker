"""Tests for routing logic."""
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from neuralbrok.router import get_vram_snapshot, route_request, RouteDecision
from neuralbrok.types import VramSnapshot
from neuralbrok.config import load_config


class TestGetVramSnapshot:
    """Tests for VRAM snapshot collection."""

    @patch.dict('sys.modules', {'pynvml': MagicMock()})
    def test_get_vram_snapshot_success(self):
        """Test successful VRAM snapshot retrieval."""
        # Mock pynvml memory info
        mock_info = MagicMock()
        mock_info.used = 2.5 * (1024**3)  # 2.5 GB
        mock_info.free = 5.5 * (1024**3)  # 5.5 GB

        import sys
        mock_pynvml = sys.modules['pynvml']
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_info
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"

        snapshot = get_vram_snapshot(gpu_id=0)

        assert snapshot.gpu_id == 0
        assert snapshot.vram_used_gb == pytest.approx(2.5, abs=0.01)
        assert snapshot.vram_free_gb == pytest.approx(5.5, abs=0.01)
        assert isinstance(snapshot.timestamp, datetime)

    def test_get_vram_snapshot_no_pynvml(self):
        """Test error when pynvml not available."""
        with patch.dict('sys.modules', {'pynvml': None}):
            with pytest.raises(RuntimeError, match="pynvml not available"):
                get_vram_snapshot()

    @patch.dict('sys.modules', {'pynvml': MagicMock()})
    def test_get_vram_snapshot_nvml_error(self):
        """Test handling of pynvml errors."""
        # Create a mock NVMLError class
        class MockNVMLError(Exception):
            pass

        import sys
        mock_pynvml = sys.modules['pynvml']
        mock_pynvml.NVMLError = MockNVMLError
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = MockNVMLError(
            "GPU not found"
        )

        with pytest.raises(RuntimeError, match="Failed to query GPU"):
            get_vram_snapshot(gpu_id=0)


class TestRouteDecision:
    """Tests for RouteDecision dataclass."""

    def test_create_decision(self):
        """Test creating a route decision."""
        vram = VramSnapshot(
            gpu_id=0,
            vram_used_gb=2.0,
            vram_free_gb=6.0,
            timestamp=datetime.now(),
        )
        decision = RouteDecision(
            backend_chosen="ollama",
            vram_at_decision=vram,
            latency_ms=5.5,
        )
        assert decision.backend_chosen == "ollama"
        assert decision.latency_ms == 5.5


class TestRouteRequest:
    """Tests for routing logic."""

    def test_route_to_ollama_high_vram(self):
        """Test routing to Ollama when VRAM is abundant."""
        config = load_config("config.yaml.example")
        vram = VramSnapshot(
            gpu_id=0,
            vram_used_gb=1.0,
            vram_free_gb=12.0,  # Above 80% of 10GB threshold
            timestamp=datetime.now(),
        )

        decision = route_request(vram, config)

        assert decision.backend_chosen == "ollama-default"
        assert decision.vram_at_decision == vram

    def test_route_to_groq_low_vram(self):
        """Test routing to Groq when VRAM is low."""
        config = load_config("config.yaml.example")
        vram = VramSnapshot(
            gpu_id=0,
            vram_used_gb=7.0,
            vram_free_gb=1.0,  # Below 4GB threshold
            timestamp=datetime.now(),
        )

        decision = route_request(vram, config)

        assert decision.backend_chosen == "groq"

    def test_route_no_backends_configured(self):
        """Test error when no backends are configured."""
        config = load_config("config.yaml.example")
        config.local_nodes = []
        config.cloud_providers = []

        vram = VramSnapshot(
            gpu_id=0,
            vram_used_gb=1.0,
            vram_free_gb=7.0,
            timestamp=datetime.now(),
        )

        with pytest.raises(ValueError, match="No backends configured"):
            route_request(vram, config)

    def test_route_respects_threshold(self):
        """Test that routing respects custom VRAM threshold."""
        config = load_config("config.yaml.example")
        config.local_nodes[0].vram_threshold_gb = 6.0  # Raise threshold

        vram = VramSnapshot(
            gpu_id=0,
            vram_used_gb=1.0,
            vram_free_gb=5.0,  # Below new 6GB threshold
            timestamp=datetime.now(),
        )

        decision = route_request(vram, config)

        # Should route to Groq because 5GB < 6GB threshold
        assert decision.backend_chosen == "groq"
