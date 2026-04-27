"""
Async background VRAM telemetry loop.
Unified cross-platform polling (NVIDIA, Apple, AMD, CPU).
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from neuralbrok.types import VramSnapshot
from neuralbrok.hardware import HardwareTelemetry

logger = logging.getLogger(__name__)

STALE_THRESHOLD_S = 2.0


class VramPoller:
    """
    Background VRAM polling loop with in-memory cached snapshot.
    Uses HardwareTelemetry for quiet, cross-platform hardware access.
    """

    def __init__(self, gpu_id: int = 0, poll_interval_s: float = 0.5):
        self._gpu_id = gpu_id
        self._poll_interval = poll_interval_s
        self._telemetry = HardwareTelemetry()
        self._snapshot: VramSnapshot = self._fallback_snapshot()
        self._task: Optional[asyncio.Task] = None
        self._initialized = False

    async def start(self) -> None:
        """Initialize telemetry and launch background polling task."""
        vendor = self._telemetry.initialize()
        self._initialized = True
        logger.info(f"Hardware telemetry active — vendor: {vendor}")
        
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Cancel polling task and shut down telemetry."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._initialized:
            self._telemetry.shutdown()
            self._initialized = False

    def latest(self) -> VramSnapshot:
        """
        Return cached VRAM snapshot. Always <1ms.
        Logs a warning if snapshot is stale (>2s old).
        """
        age_s = (datetime.now(timezone.utc) - self._snapshot.timestamp).total_seconds()
        if age_s > STALE_THRESHOLD_S:
            logger.warning(
                f"VRAM snapshot is {age_s:.1f}s old (stale) — polling may have stalled"
            )
        return self._snapshot

    async def _poll_loop(self) -> None:
        """Poll hardware on interval and update cached snapshot."""
        while True:
            try:
                stats = self._telemetry.get_vram_snapshot(self._gpu_id)
                self._snapshot = VramSnapshot(
                    gpu_id=self._gpu_id,
                    vram_used_gb=stats["used"],
                    vram_free_gb=stats["free"],
                    timestamp=datetime.now(timezone.utc),
                )
            except Exception as e:
                logger.error(f"Error in VRAM poll loop: {e}")
            await asyncio.sleep(self._poll_interval)

    def _fallback_snapshot(self) -> VramSnapshot:
        """Synthetic snapshot used before initialization."""
        return VramSnapshot(
            gpu_id=self._gpu_id,
            vram_used_gb=0.0,
            vram_free_gb=8.0,
            timestamp=datetime.now(timezone.utc),
        )
