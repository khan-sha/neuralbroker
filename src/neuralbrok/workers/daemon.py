"""
Worker Daemon: Auto-triggered background tasks (Autopilot).
Equivalent to Ruflo's loop workers.
"""
import asyncio
import logging
from typing import List, Callable, Awaitable

logger = logging.getLogger(__name__)

class WorkerDaemon:
    def __init__(self, idle_threshold_s: float = 300.0):
        self.idle_threshold_s = idle_threshold_s
        self._tasks: List[Callable[[], Awaitable[None]]] = []
        self._daemon_task = None
        self._running = False
        self._last_active_time = asyncio.get_event_loop().time()

    def register_worker(self, worker_func: Callable[[], Awaitable[None]]):
        """Register a background task to be run when idle."""
        self._tasks.append(worker_func)

    def mark_active(self):
        """Called by the main API to reset the idle timer."""
        self._last_active_time = asyncio.get_event_loop().time()

    async def start(self):
        self._running = True
        self._daemon_task = asyncio.create_task(self._loop())
        logger.info(f"Worker Daemon started with {len(self._tasks)} registered workers.")

    async def stop(self):
        self._running = False
        if self._daemon_task:
            self._daemon_task.cancel()
            try:
                await self._daemon_task
            except asyncio.CancelledError:
                pass

    async def _loop(self):
        while self._running:
            await asyncio.sleep(10.0) # Check every 10 seconds
            
            now = asyncio.get_event_loop().time()
            if now - self._last_active_time > self.idle_threshold_s:
                # System is idle, trigger workers
                logger.info("System idle threshold reached. Triggering background workers...")
                for worker in self._tasks:
                    try:
                        await worker()
                    except Exception as e:
                        logger.error(f"Background worker failed: {e}")
                
                # Reset timer so they don't run continuously without a pause
                self.mark_active()
