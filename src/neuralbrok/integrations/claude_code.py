"""
NeuralBroker Claude Code Terminal Integration (BETA)

Provides real-time routing context and model selection to Claude Code.
Enables `neuralbrok code` command to launch Claude Code with NeuralBroker awareness.

Usage:
  neuralbrok code          # Launch Claude Code with NeuralBroker routing context
  neuralbrok code --watch  # Watch routing decisions in real-time
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ClaudeCodeTerminal:
    """
    Real-time routing context provider for Claude Code terminal.
    Streams routing decisions, VRAM state, and provider health.
    """

    def __init__(self, broker_host: str = "localhost", broker_port: int = 8000):
        self.broker_url = f"http://{broker_host}:{broker_port}"
        self.session = None
        self.last_stats: Dict[str, Any] = {}
        self.last_health: Dict[str, Any] = {}

    async def connect(self):
        """Connect to NeuralBroker API."""
        import httpx
        self.session = httpx.AsyncClient(timeout=5.0)
        try:
            resp = await self.session.get(f"{self.broker_url}/health")
            resp.raise_for_status()
            logger.info("Connected to NeuralBroker")
        except Exception as e:
            logger.error(f"Failed to connect to NeuralBroker: {e}")
            return False
        return True

    async def disconnect(self):
        """Close connection."""
        if self.session:
            await self.session.aclose()

    async def get_routing_context(self) -> Dict[str, Any]:
        """Fetch current routing state and health."""
        if not self.session:
            return {}

        try:
            stats_resp = await self.session.get(f"{self.broker_url}/nb/stats")
            health_resp = await self.session.get(f"{self.broker_url}/health")

            stats_resp.raise_for_status()
            health_resp.raise_for_status()

            stats = stats_resp.json()
            health = health_resp.json()

            return {
                "timestamp": datetime.now().isoformat(),
                "health": health,
                "stats": stats,
                "connected": True,
            }
        except Exception as e:
            logger.error(f"Error fetching routing context: {e}")
            return {"connected": False, "error": str(e)}

    async def stream_routing_decisions(self, callback: Optional[callable] = None):
        """
        Stream routing decisions in real-time.
        Polls /nb/stats every 500ms and calls callback with updates.
        """
        if not self.session:
            return

        while True:
            try:
                context = await self.get_routing_context()
                if callback:
                    callback(context)
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(1.0)

    def format_context(self, context: Dict[str, Any]) -> str:
        """Format routing context as readable text for terminal."""
        if not context.get("connected"):
            return f"[ERROR] {context.get('error', 'Not connected to NeuralBroker')}"

        health = context.get("health", {})
        stats = context.get("stats", {})

        lines = [
            f"NeuralBroker Status · {context['timestamp'][-8:]}",
            f"  Uptime: {health.get('uptime_seconds', 'N/A')}s",
            f"  Requests: {stats.get('total_requests', 0)} total",
            f"  Mode: {stats.get('routing_mode', 'unknown')}",
        ]

        if "smart_classifications" in stats:
            lines.append(f"  Smart Classifications: {stats['smart_classifications']}")

        if "avg_classify_ms" in stats:
            lines.append(f"  Avg Classify Time: {stats['avg_classify_ms']:.1f}ms")

        return "\n".join(lines)


async def launch_code_with_routing_context(
    broker_host: str = "localhost",
    broker_port: int = 8000,
    watch: bool = False,
):
    """
    Launch Claude Code with NeuralBroker routing context.

    Args:
        broker_host: NeuralBroker host (default: localhost)
        broker_port: NeuralBroker port (default: 8000)
        watch: Stream routing decisions in real-time
    """
    terminal = ClaudeCodeTerminal(broker_host, broker_port)

    if not await terminal.connect():
        print("ERROR: Could not connect to NeuralBroker")
        print(f"Make sure NeuralBroker is running: neuralbrok start")
        return

    print(f"\n  Connected to NeuralBroker at {terminal.broker_url}\n")

    if watch:
        print("  Streaming routing decisions (Ctrl+C to stop)...\n")

        async def on_update(context):
            print(f"\r{terminal.format_context(context)}", end="", flush=True)

        try:
            await terminal.stream_routing_decisions(on_update)
        except KeyboardInterrupt:
            print("\n\nStopped.")
    else:
        # One-shot context display
        context = await terminal.get_routing_context()
        print(terminal.format_context(context))
        print(f"\n  Use --watch flag to stream in real-time")

    await terminal.disconnect()
