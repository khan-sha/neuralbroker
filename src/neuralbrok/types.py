"""
Type definitions for OpenAI-compatible API, routing, and policy engine.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Policy Modes ──────────────────────────────────────────────────────────────

class PolicyMode(str, Enum):
    """Routing policy modes."""
    COST = "cost"
    SPEED = "speed"
    FALLBACK = "fallback"


# ── OpenAI Wire Types ────────────────────────────────────────────────────────

class OpenAIRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: list[dict]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1)
    stream: Optional[bool] = Field(default=False)
    # Preserve extra fields (tools, tool_choice, etc.) for pass-through
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str] | str] = None
    n: Optional[int] = None
    user: Optional[str] = None


class OpenAIResponse(BaseModel):
    """Simplified OpenAI response for streaming chunks."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list


class RoutingMetadata(BaseModel):
    """Metadata about routing decision."""

    backend_chosen: str
    vram_used_gb: float
    vram_free_gb: float
    latency_ms: float


# ── VRAM Telemetry ────────────────────────────────────────────────────────────

@dataclass
class VramSnapshot:
    """GPU VRAM state at a point in time."""

    gpu_id: int
    vram_used_gb: float
    vram_free_gb: float
    timestamp: datetime


@dataclass
class GpuState:
    """Per-GPU state including utilization ratio."""

    gpu_index: int
    used_gb: float
    total_gb: float
    utilization: float  # 0.0 - 1.0
    available: bool


# ── Routing Decisions ─────────────────────────────────────────────────────────

@dataclass
class RouteDecision:
    """Result of a routing decision from the policy engine."""

    backend_chosen: str
    fallback_chain: list[str] = field(default_factory=list)
    vram_at_decision: Optional[VramSnapshot] = None
    policy_mode: PolicyMode = PolicyMode.COST
    latency_ms: float = 0.0
    reason: str = ""  # "vram_ok" | "vram_full" | "speed_mode" | "fallback"
    cost_usd: float = 0.0
    dollars_saved: float = 0.0
    timestamp: Optional[datetime] = None


# ── Provider Status ───────────────────────────────────────────────────────────

@dataclass
class CircuitState:
    """Circuit breaker state for a provider."""

    consecutive_errors: int = 0
    failed: bool = False
    failed_at: Optional[datetime] = None
    recovery_seconds: float = 60.0


@dataclass
class ProviderStatus:
    """Health and circuit breaker status for a provider."""

    name: str
    provider_type: str  # "local" | "cloud"
    healthy: bool = True
    circuit: CircuitState = field(default_factory=CircuitState)
    requests_total: int = 0
    errors_total: int = 0
    last_latency_ms: float = 0.0





