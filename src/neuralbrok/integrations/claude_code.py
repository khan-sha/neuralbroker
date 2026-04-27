"""
NeuralBroker Claude Code Terminal Integration

Provides real-time routing context and model selection to Claude Code.
Enables `neuralbrok code` command to launch Claude Code with NeuralBroker awareness.

Usage:
  neuralbrok code          # Show routing context then launch Claude Code via NeuralBroker
  neuralbrok code --watch  # Stream routing decisions while Claude Code runs
"""

import asyncio
import json
import logging
import os
import subprocess
import threading
from datetime import datetime, timezone
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
                "timestamp": datetime.now(timezone.utc).isoformat(),
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


def build_routing_env(nb_url: str, api_key: str = "nb-local") -> dict:
    """Build env vars that route Claude Code through NeuralBroker."""
    return {
        "ANTHROPIC_BASE_URL": f"{nb_url}/v1",
        "OPENAI_BASE_URL": f"{nb_url}/v1",
        "OPENAI_API_KEY": api_key,
    }


def launch_claude(env: dict) -> int:
    """
    Launch the Claude Code CLI with NeuralBroker routing env vars.
    Returns the process exit code.
    """
    merged_env = {**os.environ, **env}
    try:
        result = subprocess.run(["claude"], env=merged_env)
        return result.returncode
    except FileNotFoundError:
        print("\n  claude CLI not found in PATH.")
        print("  Install Claude Code: https://claude.ai/code")
        print("  Or: npm install -g @anthropic-ai/claude-code")
        return 1


async def launch_code_with_routing_context(
    broker_host: str = "localhost",
    broker_port: int = 8000,
    watch: bool = False,
):
    """
    Launch Claude Code with NeuralBroker routing context.

    1. Connect to NeuralBroker and display current routing state.
    2. Set env vars so Claude Code routes through NeuralBroker.
    3. If --watch: stream routing decisions in a background thread while Claude Code runs.
    4. Launch the claude CLI subprocess.
    """
    terminal = ClaudeCodeTerminal(broker_host, broker_port)

    if not await terminal.connect():
        print("  ERROR: Could not connect to NeuralBroker")
        print(f"  Make sure NeuralBroker is running: neuralbrok start")
        return

    # Show current routing context before launch
    context = await terminal.get_routing_context()
    print(f"\n{terminal.format_context(context)}\n")

    nb_url = terminal.broker_url
    api_key = os.environ.get("NB_API_KEY", "nb-local")
    env = build_routing_env(nb_url, api_key)

    print(f"  Routing: ANTHROPIC_BASE_URL={nb_url}/v1")
    print(f"  Launching Claude Code...\n")

    await terminal.disconnect()

    if watch:
        # Stream routing stats in background while Claude Code runs
        stop_event = threading.Event()

        def _watch_loop():
            import httpx
            while not stop_event.is_set():
                try:
                    with httpx.Client(timeout=2.0) as c:
                        r = c.get(f"{nb_url}/nb/stats")
                        if r.status_code == 200:
                            s = r.json()
                            total = s.get("total_requests", 0)
                            local_pct = s.get("local_pct", 0)
                            saved = s.get("total_saved", 0.0)
                            print(
                                f"\r  [NB] reqs={total}  local={local_pct}%  saved=${saved:.4f}",
                                end="",
                                flush=True,
                            )
                except Exception:
                    pass
                stop_event.wait(0.5)

        watcher = threading.Thread(target=_watch_loop, daemon=True)
        watcher.start()
        try:
            launch_claude(env)
        finally:
            stop_event.set()
            print()  # newline after the watch line
    else:
        launch_claude(env)
