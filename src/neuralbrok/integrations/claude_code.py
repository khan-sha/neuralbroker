"""
NeuralBroker ↔ Claude Code terminal integration.

`neuralbrok code` launches the Claude Code CLI (`claude`) with NeuralBroker
acting as the OpenAI-compatible proxy, so every Claude Code request is routed
through NeuralBroker's VRAM-aware engine just like any other OpenAI API call.

Usage:
  neuralbrok code          # launch Claude Code backed by NeuralBroker
  neuralbrok code --watch  # same, plus stream live routing decisions
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

PINK   = "\033[38;5;213m"
MATRIX = "\033[38;5;82m"
CYAN   = "\033[38;5;51m"
DIM    = "\033[2m"
RED    = "\033[38;5;203m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


async def _check_broker(broker_url: str) -> bool:
    """Return True if NeuralBroker is reachable."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{broker_url}/health")
            return r.status_code == 200
    except Exception:
        return False


async def _get_context(broker_url: str) -> Dict[str, Any]:
    """Fetch live routing state from NeuralBroker."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            stats  = (await c.get(f"{broker_url}/nb/stats")).json()
            health = (await c.get(f"{broker_url}/health")).json()
            vram_d = (await c.get(f"{broker_url}/nb/vram")).json()
        vram_pct = 0
        for v in vram_d.values():
            vram_pct = int(v.get("utilization", 0) * 100)
            break
        return {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "health": health,
            "stats": stats,
            "vram_pct": vram_pct,
            "connected": True,
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


def _format_status(ctx: Dict[str, Any]) -> str:
    if not ctx.get("connected"):
        return f"  {RED}✗ NeuralBroker: {ctx.get('error', 'disconnected')}{RESET}"
    h = ctx["health"]
    s = ctx["stats"]
    mode   = h.get("mode", "?")
    total  = s.get("total_requests", 0)
    local_pct = s.get("local_pct", 0)
    saved  = s.get("total_saved", 0.0)
    vram   = ctx["vram_pct"]
    ts     = ctx["timestamp"]
    return (
        f"  {MATRIX}●{RESET}  NeuralBroker {DIM}{ts}{RESET}  "
        f"mode={PINK}{mode}{RESET}  "
        f"req={total}  local={local_pct}%  "
        f"vram={vram}%  saved=${saved:.4f}"
    )


async def launch_code_with_routing_context(
    broker_host: str = "localhost",
    broker_port: int = 8000,
    watch: bool = False,
    api_key: str = "neuralbrok",
):
    """
    Main entry point called by `neuralbrok code`.

    1. Verify NeuralBroker is running.
    2. If the `claude` CLI is available, launch it with ANTHROPIC_BASE_URL
       pointing at NeuralBroker's /v1 proxy so every request is VRAM-routed.
    3. If `claude` is not installed, print setup instructions and exit.
    4. If --watch is passed, stream a live status panel alongside the shell.
    """
    broker_url = f"http://{broker_host}:{broker_port}"

    # ── 1. Verify broker is up ────────────────────────────────────────────
    if not await _check_broker(broker_url):
        print(f"\n  {RED}✗ NeuralBroker is not running at {broker_url}{RESET}")
        print(f"  {DIM}Start it first:  neuralbrok start{RESET}\n")
        return

    ctx = await _get_context(broker_url)
    print(_format_status(ctx))
    print()

    # ── 2. Check for claude CLI ───────────────────────────────────────────
    claude_bin = shutil.which("claude")
    if claude_bin is None:
        print(f"  {RED}✗ `claude` CLI not found in PATH{RESET}")
        print(f"  {DIM}Install Claude Code: https://claude.ai/code{RESET}")
        print()
        print(f"  {PINK}Manual setup{RESET} — set these env vars before running `claude`:")
        print(f"  {DIM}  export ANTHROPIC_BASE_URL={broker_url}/v1{RESET}")
        print(f"  {DIM}  export ANTHROPIC_API_KEY={api_key}{RESET}")
        print(f"  {DIM}  claude{RESET}")
        return

    # ── 3. Build env for claude CLI ───────────────────────────────────────
    env = {**os.environ}
    env["ANTHROPIC_BASE_URL"] = f"{broker_url}/v1"
    # Only override the key if none is set — NB accepts any key when NB_API_KEY is unset.
    if not env.get("ANTHROPIC_API_KEY"):
        env["ANTHROPIC_API_KEY"] = api_key

    print(f"  {MATRIX}✓{RESET}  Launching Claude Code via NeuralBroker")
    print(f"  {DIM}  ANTHROPIC_BASE_URL={broker_url}/v1{RESET}")
    print(f"  {DIM}  All requests routed through NeuralBroker's policy engine{RESET}")
    print()

    if watch:
        # Run claude in background and stream status alongside it
        proc = subprocess.Popen([claude_bin], env=env)

        async def _stream():
            while proc.poll() is None:
                c = await _get_context(broker_url)
                sys.stdout.write(f"\r{_format_status(c)}  ")
                sys.stdout.flush()
                await asyncio.sleep(1.0)

        try:
            await _stream()
        except asyncio.CancelledError:
            pass
        finally:
            if proc.poll() is None:
                proc.terminate()
    else:
        # Exec replaces the current process — cleaner UX
        os.execve(claude_bin, [claude_bin], env)
