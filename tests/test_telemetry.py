import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from neuralbrok.telemetry import VramPoller, STALE_THRESHOLD_S
from neuralbrok.types import VramSnapshot

class TestVramPollerLifecycle:

    @pytest.mark.asyncio
    async def test_start_creates_background_task(self):
        poller = VramPoller(poll_interval_s=60)
        with patch("neuralbrok.hardware.HardwareTelemetry.initialize"):
            await poller.start()
            assert poller._task is not None
            assert not poller._task.done()
            await poller.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        poller = VramPoller(poll_interval_s=60)
        with patch("neuralbrok.hardware.HardwareTelemetry.initialize"):
            await poller.start()
            await poller.stop()
            assert poller._task.done()

class TestVramPollerFallback:
    @pytest.mark.asyncio
    async def test_fallback_snapshot_when_pynvml_unavailable(self):
        poller = VramPoller(poll_interval_s=60)
        with patch("neuralbrok.hardware.HardwareTelemetry.initialize", side_effect=Exception("No driver")):
            with pytest.raises(Exception):
                await poller.start()

class TestVramPollerLive:
    @pytest.mark.asyncio
    async def test_snapshot_reflects_gpu_memory(self):
        with patch("neuralbrok.hardware.HardwareTelemetry.initialize", return_value="nvidia"):
            with patch("neuralbrok.hardware.HardwareTelemetry.get_vram_snapshot", return_value={"used": 6.0, "free": 4.0}):
                poller = VramPoller(poll_interval_s=0.05)
                await poller.start()

                await asyncio.sleep(0.15)

                snap = poller.latest()
                assert abs(snap.vram_used_gb - 6.0) < 0.01
                assert abs(snap.vram_free_gb - 4.0) < 0.01

                await poller.stop()

class TestVramPollerStale:
    @pytest.mark.asyncio
    async def test_stale_snapshot_logs_warning(self, caplog):
        import logging
        with patch("neuralbrok.hardware.HardwareTelemetry.initialize", return_value="nvidia"):
            poller = VramPoller(poll_interval_s=60)
            await poller.start()

            old_ts = datetime.now() - timedelta(seconds=STALE_THRESHOLD_S + 1)
            poller._snapshot = VramSnapshot(
                gpu_id=0, vram_used_gb=0.0, vram_free_gb=8.0, timestamp=old_ts
            )

            with caplog.at_level(logging.WARNING, logger="neuralbrok.telemetry"):
                poller.latest()

            assert any("stale" in r.message.lower() for r in caplog.records)
            await poller.stop()
