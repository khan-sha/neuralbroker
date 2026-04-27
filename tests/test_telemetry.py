"""
Tests for async VRAM telemetry loop.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from neuralbrok.telemetry import VramPoller, STALE_THRESHOLD_S
from neuralbrok.types import VramSnapshot


# ── helpers ──────────────────────────────────────────────────────────────────

def make_nvml_mock(used_bytes: int, free_bytes: int):
    """Build a pynvml mock that returns specific memory values."""
    mock = MagicMock()
    mem_info = MagicMock()
    mem_info.used = used_bytes
    mem_info.free = free_bytes
    mock.nvmlDeviceGetMemoryInfo.return_value = mem_info
    mock.NVMLError = Exception
    return mock


GB = 1024 ** 3


# ── start / stop ──────────────────────────────────────────────────────────────

class TestVramPollerLifecycle:

    @pytest.mark.asyncio
    async def test_start_creates_background_task(self):
        poller = VramPoller(poll_interval_s=60)
        await poller.start()
        assert poller._task is not None
        assert not poller._task.done()
        await poller.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        poller = VramPoller(poll_interval_s=60)
        await poller.start()
        await poller.stop()
        assert poller._task.done()

    @pytest.mark.asyncio
    async def test_stop_idempotent_when_not_started(self):
        poller = VramPoller()
        await poller.stop()  # should not raise


# ── fallback behavior (no GPU) ────────────────────────────────────────────────

class TestVramPollerFallback:

    @pytest.mark.asyncio
    async def test_fallback_snapshot_when_pynvml_unavailable(self):
        poller = VramPoller(poll_interval_s=60)
        # _nvml_initialized stays False (no real GPU in CI)
        await poller.start()

        snap = poller.latest()
        assert isinstance(snap, VramSnapshot)
        assert snap.vram_free_gb == 8.0
        assert snap.vram_used_gb == 0.0
        assert snap.gpu_id == 0

        await poller.stop()

    @pytest.mark.asyncio
    async def test_fallback_snapshot_on_nvml_init_failure(self):
        with patch("neuralbrok.telemetry.HAS_PYNVML", True):
            mock_pynvml = MagicMock()
            mock_pynvml.nvmlInit.side_effect = Exception("no driver")
            mock_pynvml.NVMLError = Exception
            with patch("neuralbrok.telemetry.pynvml", mock_pynvml):
                poller = VramPoller(poll_interval_s=60)
                await poller.start()

                snap = poller.latest()
                assert snap.vram_free_gb == 8.0  # fallback

                await poller.stop()


# ── live snapshot (mocked pynvml) ─────────────────────────────────────────────

class TestVramPollerLive:

    @pytest.mark.asyncio
    async def test_snapshot_reflects_gpu_memory(self):
        used = int(6 * GB)
        free = int(4 * GB)

        with patch("neuralbrok.telemetry.HAS_PYNVML", True):
            mock_pynvml = make_nvml_mock(used, free)
            with patch("neuralbrok.telemetry.pynvml", mock_pynvml):
                poller = VramPoller(poll_interval_s=0.05)
                poller._nvml_initialized = True  # skip real nvmlInit
                await poller.start()

                await asyncio.sleep(0.15)  # let loop tick 2-3x

                snap = poller.latest()
                assert abs(snap.vram_used_gb - 6.0) < 0.01
                assert abs(snap.vram_free_gb - 4.0) < 0.01

                await poller.stop()

    @pytest.mark.asyncio
    async def test_latest_does_not_call_pynvml(self):
        with patch("neuralbrok.telemetry.HAS_PYNVML", True):
            mock_pynvml = make_nvml_mock(int(4 * GB), int(6 * GB))
            with patch("neuralbrok.telemetry.pynvml", mock_pynvml):
                poller = VramPoller(poll_interval_s=60)
                poller._nvml_initialized = True
                await poller.start()

                call_count_before = mock_pynvml.nvmlDeviceGetMemoryInfo.call_count
                for _ in range(100):
                    poller.latest()
                call_count_after = mock_pynvml.nvmlDeviceGetMemoryInfo.call_count

                # latest() must not call pynvml — only the background task does
                assert call_count_after == call_count_before

                await poller.stop()


# ── stale snapshot warning ────────────────────────────────────────────────────

class TestVramPollerStale:

    @pytest.mark.asyncio
    async def test_stale_snapshot_logs_warning(self, caplog):
        import logging
        poller = VramPoller(poll_interval_s=60)
        await poller.start()

        # Backdate the cached snapshot
        old_ts = datetime.now(timezone.utc) - timedelta(seconds=STALE_THRESHOLD_S + 1)
        poller._snapshot = VramSnapshot(
            gpu_id=0, vram_used_gb=0.0, vram_free_gb=8.0, timestamp=old_ts
        )

        with caplog.at_level(logging.WARNING, logger="neuralbrok.telemetry"):
            poller.latest()

        assert any("stale" in r.message.lower() or "old" in r.message.lower()
                   for r in caplog.records)

        await poller.stop()

    @pytest.mark.asyncio
    async def test_fresh_snapshot_no_warning(self, caplog):
        import logging
        poller = VramPoller(poll_interval_s=60)
        await poller.start()

        with caplog.at_level(logging.WARNING, logger="neuralbrok.telemetry"):
            poller.latest()

        stale_warnings = [r for r in caplog.records
                          if "stale" in r.message.lower() or "old" in r.message.lower()]
        assert len(stale_warnings) == 0

        await poller.stop()
