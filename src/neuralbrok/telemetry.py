"""
Async background VRAM telemetry loop.
Polls GPU memory every 500ms, caches latest snapshot for <1ms access on request path.
"""
import asyncio
import logging
import warnings
from datetime import datetime
from typing import Optional

from neuralbrok.types import VramSnapshot

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*deprecated.*")
        import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

logger = logging.getLogger(__name__)

STALE_THRESHOLD_S = 2.0


class VramPoller:
    """
    Background VRAM polling loop with in-memory cached snapshot.

    Calls nvmlInit once at startup (not per-poll), polls every poll_interval_s,
    stores latest snapshot. request path calls latest() — no pynvml call, <1ms.
    """

    def __init__(self, gpu_id: int = 0, poll_interval_s: float = 0.5):
        self._gpu_id = gpu_id
        self._poll_interval = poll_interval_s
        self._snapshot: VramSnapshot = self._fallback_snapshot()
        self._task: Optional[asyncio.Task] = None
        self._nvml_initialized = False

    async def start(self) -> None:
        """Initialize pynvml and launch background polling task."""
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
                logger.info("pynvml initialized — live VRAM polling active")
            except pynvml.NVMLError as e:
                logger.warning(f"pynvml init failed, using fallback snapshot: {e}")
        else:
            logger.warning("pynvml not installed — using fallback snapshot (vram_free=8.0GB)")

        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Cancel polling task and shut down pynvml."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
            except Exception as e:
                logger.warning(f"nvmlShutdown failed: {e}")

    def latest(self) -> VramSnapshot:
        """
        Return cached VRAM snapshot. Never calls pynvml. Always <1ms.
        Logs a warning if snapshot is stale (>2s old).
        """
        age_s = (datetime.now() - self._snapshot.timestamp).total_seconds()
        if age_s > STALE_THRESHOLD_S:
            logger.warning(
                f"VRAM snapshot is {age_s:.1f}s old (threshold {STALE_THRESHOLD_S}s) "
                "— polling may have stalled"
            )
        return self._snapshot

    async def _poll_loop(self) -> None:
        """Poll GPU on interval and update cached snapshot."""
        while True:
            try:
                self._snapshot = self._take_snapshot()
            except Exception as e:
                logger.error(f"Unexpected error in poll loop: {e}")
            await asyncio.sleep(self._poll_interval)

    def _take_snapshot(self) -> VramSnapshot:
        """Query pynvml for current VRAM state, or return fallback."""
        if not self._nvml_initialized:
            return self._fallback_snapshot()

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return VramSnapshot(
                gpu_id=self._gpu_id,
                vram_used_gb=info.used / (1024 ** 3),
                vram_free_gb=info.free / (1024 ** 3),
                timestamp=datetime.now(),
            )
        except pynvml.NVMLError as e:
            logger.error(f"VRAM query failed for gpu_id={self._gpu_id}: {e}")
            return self._fallback_snapshot()

    def _fallback_snapshot(self) -> VramSnapshot:
        """Synthetic snapshot used when GPU unavailable (non-NVIDIA / no driver)."""
        return VramSnapshot(
            gpu_id=self._gpu_id,
            vram_used_gb=0.0,
            vram_free_gb=8.0,
            timestamp=datetime.now(),
        )
