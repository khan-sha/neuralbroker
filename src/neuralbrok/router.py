"""
Policy engine with 3 routing modes, provider scoring, and circuit breakers.

Replaces the simple threshold router from Week 1.
"""
import json
import asyncio
import httpx
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from neuralbrok.config import Config
from neuralbrok.types import (
    PolicyMode,
    RouteDecision,
    VramSnapshot,
    CircuitState,
    ProviderStatus,
)
from neuralbrok.selector import SmartModelSelector
from neuralbrok.models import get_runnable_models, resolve_model
from neuralbrok.detect import detect_device

logger = logging.getLogger(__name__)

# Circuit breaker constants
CIRCUIT_ERROR_THRESHOLD = 3
CIRCUIT_RECOVERY_SECONDS = 60.0
FALLBACK_VRAM_SPILL_THRESHOLD = 0.95
FALLBACK_VRAM_RECOVER_THRESHOLD = 0.85


@dataclass
class ProviderScore:
    """Scored provider for routing decision."""
    name: str
    provider_type: str  # "local" | "cloud"
    score: float
    cost_per_1k: float
    failed: bool = False
    reason: str = ""


class PolicyEngine:
    """VRAM-aware routing policy engine.

    Implements cost-mode, speed-mode, and fallback-mode routing
    with per-provider circuit breakers and scoring.
    """

    def __init__(self, config: Config):
        self.config = config
        self.mode = PolicyMode(config.routing.default_mode)

        # Circuit breaker state per provider name
        self._circuits: dict[str, CircuitState] = {}

        # Latency tracking
        self._latencies: dict[str, deque] = {}

        # Routing decision log (last 500)
        self._routing_log: deque[dict] = deque(maxlen=500)

        # Aggregate stats
        self._stats = {
            "total_requests": 0,
            "local_requests": 0,
            "cloud_requests": 0,
            "total_cost_local": 0.0,
            "total_cost_cloud": 0.0,
            "total_saved": 0.0,
            "smart_classifications": 0,
            "total_classify_ms": 0.0,
            "fallback_to_cost_count": 0,
            "session_start": datetime.now(timezone.utc).isoformat(),
        }

        # Fallback mode: track whether we're in "spill to cloud" state
        self._fallback_spilling = False

        # Track last error per provider for fallback mode
        self._last_error: dict[str, datetime] = {}

    # Cache hardware profile to avoid expensive detection on every request
    _cached_hw_profile = None

    async def decide_async(
        self,
        request_body: dict,
        vram: Optional[VramSnapshot],
        available_providers: list[str],
        provider_types: dict[str, str],
        provider_costs: dict[str, float],
        requested_model: str = "",
    ) -> RouteDecision:
        if self.mode == PolicyMode.SMART:
            try:
                if not self._cached_hw_profile:
                    self._cached_hw_profile = detect_device()
                prof = self._cached_hw_profile
                
                device_key = prof.gpu_model if prof.gpu_vendor != "none" else prof.platform
                vram_total = vram.vram_used_gb + vram.vram_free_gb if vram else 0
                runnable = get_runnable_models(vram_total, prof.ram_gb, device_key)
                
                # If user defined an allowed pool, restrict to it
                allowed = getattr(self.config, "allowed_models", [])
                if not allowed and self.config.routing:
                    allowed = getattr(self.config.routing, "allowed_models", [])
                
                if allowed:
                    runnable = [m for m in runnable if m.name in allowed]
                
                # ── Step 1: Prompt Classification ──
                # Use a small local model for ultra-fast classification
                small_model = resolve_model("small")
                mini_model = resolve_model("phi-4-mini")
                classifier_model = small_model if any(m.name == small_model for m in runnable) else mini_model
                
                start_class = time.perf_counter()
                prompt_text = ""
                if "messages" in request_body and request_body["messages"]:
                    prompt_text = str(request_body["messages"][-1].get("content", ""))[:300]
                
                system_prompt = 'Classify this prompt into categories. Respond with only a JSON array of strings.\nCategories: ["chat", "coding", "math", "reasoning", "vision", "tools", "rag", "multilingual"]'
                
                # Check what's actually installed in Ollama
                loaded_models = []
                try:
                    async with httpx.AsyncClient(timeout=0.8) as client:
                        r = await client.get("http://localhost:11434/api/tags")
                        if r.status_code == 200:
                            loaded_models = [m["name"] for m in r.json().get("models", [])]
                except Exception:
                    pass
                
                categories = []
                class_time = 0
                try:
                    async with httpx.AsyncClient(timeout=2.0) as client:
                        resp = await client.post("http://localhost:11434/api/chat", json={
                            "model": classifier_model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt_text}
                            ],
                            "stream": False
                        })
                        if resp.status_code == 200:
                            content = resp.json()["message"]["content"]
                            import re
                            match = re.search(r'\[.*\]', content, re.DOTALL)
                            categories = json.loads(match.group(0)) if match else []
                            class_time = int((time.perf_counter() - start_class) * 1000)
                except Exception:
                    categories = ["chat"] # Default fallback

                # ── Step 2: Quality-Aware Selection ──
                selector = SmartModelSelector(device_key, vram.vram_free_gb if vram else 0, runnable, hw_profile=prof)
                best_local_models = selector.for_workload(categories)
                
                # Check if we have any of the best local models installed
                chosen_local = None
                for bm in best_local_models:
                    if bm.name in loaded_models or f"{bm.name}:latest" in loaded_models:
                        chosen_local = bm
                        break
                
                # AI Level logic: If the task is "reasoning" or "coding" and local model is small,
                # we should consider cloud if available.
                needs_frontier = any(c in categories for c in ["reasoning", "coding", "math"])
                local_is_small = chosen_local and chosen_local.params_b < 14 # < 14B is "standard"
                
                # ── Step 3: Cloud Fallback Check ──
                cloud_tags = getattr(self.config, "ollama_cloud_models", [])
                if not cloud_tags:
                    # Direct check for config key
                    cloud_tags = getattr(self.config.routing, "ollama_cloud_models", [])

                should_route_cloud = False
                cloud_reason = ""
                
                if needs_frontier and local_is_small and cloud_tags:
                    should_route_cloud = True
                    cloud_reason = "frontier_task_vs_small_local"
                elif not chosen_local and cloud_tags:
                    should_route_cloud = True
                    cloud_reason = "no_suitable_local_installed"
                elif vram and vram.vram_free_gb < 1.0 and cloud_tags: # Near OOM
                    should_route_cloud = True
                    cloud_reason = "vram_critically_low"

                if should_route_cloud:
                    cloud_model = cloud_tags[0]
                    request_body["model"] = cloud_model
                    latency_ms = (time.perf_counter() - start_class) * 1000
                    logger.info(f"[smart→cloud] {cloud_reason} → {cloud_model}")
                    
                    local_prov = next((p for p, pt in provider_types.items() if pt == "local"), "ollama")
                    decision = RouteDecision(
                        backend_chosen=local_prov,
                        fallback_chain=[p for p in available_providers if p != local_prov],
                        vram_at_decision=vram,
                        policy_mode=self.mode,
                        classified_as=",".join(categories) + ":cloud",
                        latency_ms=latency_ms,
                        reason=f"smart:cloud:{cloud_reason}:{cloud_model}",
                        timestamp=datetime.now(timezone.utc)
                    )
                    self._routing_log.append({
                        "backend": local_prov,
                        "mode": "smart",
                        "reason": f"smart→cloud: {cloud_reason} → {cloud_model}",
                        "classified_as": categories,
                        "chosen_model": cloud_model,
                        "is_cloud": True,
                        "latency_ms": round(latency_ms, 2),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    self._stats["total_requests"] += 1
                    self._stats["cloud_requests"] += 1
                    return decision

                # ── Step 4: Final Local Decision ──
                if not chosen_local and loaded_models:
                    chosen_local_name = loaded_models[0].replace(":latest", "")
                elif chosen_local:
                    chosen_local_name = chosen_local.name
                else:
                    chosen_local_name = resolve_model("default")

                request_body["model"] = chosen_local_name
                latency_ms = (time.perf_counter() - start_class) * 1000
                logger.info(f"[smart] {chosen_local_name} · categories: {categories}")
                
                local_prov = next((p for p, pt in provider_types.items() if pt == "local"), "ollama")
                decision = RouteDecision(
                    backend_chosen=local_prov,
                    fallback_chain=[p for p in available_providers if p != local_prov],
                    vram_at_decision=vram,
                    policy_mode=self.mode,
                    classified_as=",".join(categories),
                    latency_ms=latency_ms,
                    reason=f"smart:local:{chosen_local_name}",
                    timestamp=datetime.now(timezone.utc)
                )
                self._routing_log.append({
                    "backend": local_prov,
                    "mode": "smart",
                    "reason": f"smart:local → {chosen_local_name}",
                    "classified_as": categories,
                    "chosen_model": chosen_local_name,
                    "latency_ms": round(latency_ms, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                self._stats["total_requests"] += 1
                self._stats["local_requests"] += 1
                self._stats["smart_classifications"] += 1
                self._stats["total_classify_ms"] += class_time
                return decision

            except Exception as e:
                logger.warning(f"Smart route failed, falling back to cost mode: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                self._stats["fallback_to_cost_count"] += 1
        
        return self.decide(vram, available_providers, provider_types, provider_costs, requested_model)

    def decide(
        self,
        vram: Optional[VramSnapshot],
        available_providers: list[str],
        provider_types: dict[str, str],
        provider_costs: dict[str, float],
        requested_model: str = "",
    ) -> RouteDecision:
        """Make a routing decision based on current state.

        Args:
            vram: Current VRAM snapshot (may be None if no GPU).
            available_providers: List of provider names that are configured.
            provider_types: Map of provider name → "local" | "cloud".
            provider_costs: Map of provider name → cost_per_1k_tokens.

        Returns:
            RouteDecision with chosen backend and fallback chain.
        """
        start = time.perf_counter()
        self._stats["total_requests"] += 1

        vram_util = 0.0
        if vram and (vram.vram_used_gb + vram.vram_free_gb) > 0:
            total = vram.vram_used_gb + vram.vram_free_gb
            vram_util = vram.vram_used_gb / total

        # Filter out circuit-broken providers
        active = []
        for name in available_providers:
            circuit = self._get_circuit(name)
            if circuit.failed:
                # Check if recovery period has elapsed
                if (circuit.failed_at and
                        datetime.now(timezone.utc) - circuit.failed_at
                        > timedelta(seconds=circuit.recovery_seconds)):
                    circuit.failed = False
                    circuit.consecutive_errors = 0
                    logger.info(f"Circuit breaker recovered for {name}")
                else:
                    continue
            active.append(name)

        if not active:
            # All providers circuit-broken — try to recover the most recent
            for name in available_providers:
                circuit = self._get_circuit(name)
                circuit.failed = False
                circuit.consecutive_errors = 0
                active.append(name)
            logger.warning("All providers circuit-broken — forcing recovery")

        # Score providers based on policy mode
        scored = self._score_providers(
            active, provider_types, provider_costs, vram_util,
            requested_model=requested_model,
        )

        # Sort by score descending
        scored.sort(key=lambda s: s.score, reverse=True)

        # Pick winner and build fallback chain
        winner = scored[0] if scored else None
        fallback_chain = [s.name for s in scored[1:] if not s.failed]

        reason = self._explain_decision(winner, vram_util)
        latency_ms = (time.perf_counter() - start) * 1000

        decision = RouteDecision(
            backend_chosen=winner.name if winner else "",
            fallback_chain=fallback_chain,
            vram_at_decision=vram,
            policy_mode=self.mode,
            latency_ms=latency_ms,
            reason=reason,
            timestamp=datetime.now(timezone.utc),
        )

        # Log the decision
        log_entry = {
            "backend": decision.backend_chosen,
            "mode": self.mode.value,
            "reason": reason,
            "vram_util": round(vram_util, 3),
            "latency_ms": round(latency_ms, 2),
            "fallback_chain": fallback_chain,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._routing_log.append(log_entry)

        # Track local vs cloud
        if winner and provider_types.get(winner.name) == "local":
            self._stats["local_requests"] += 1
        elif winner:
            self._stats["cloud_requests"] += 1

        return decision

    def record_success(self, provider_name: str) -> None:
        """Record a successful request — reset circuit breaker."""
        circuit = self._get_circuit(provider_name)
        circuit.consecutive_errors = 0
        circuit.failed = False

    def record_error(self, provider_name: str) -> None:
        """Record a failed request — increment circuit breaker."""
        circuit = self._get_circuit(provider_name)
        circuit.consecutive_errors += 1
        self._last_error[provider_name] = datetime.now(timezone.utc)

        if circuit.consecutive_errors >= CIRCUIT_ERROR_THRESHOLD:
            circuit.failed = True
            circuit.failed_at = datetime.now(timezone.utc)
            logger.warning(
                f"Circuit breaker OPEN for {provider_name} "
                f"({circuit.consecutive_errors} consecutive errors)"
            )

    def record_cost(
        self,
        provider_name: str,
        provider_type: str,
        cost_usd: float,
        cheapest_cloud_cost: float,
    ) -> None:
        """Record cost for a completed request and compute savings."""
        if provider_type == "local":
            self._stats["total_cost_local"] += cost_usd
            saved = max(0, cheapest_cloud_cost - cost_usd)
            self._stats["total_saved"] += saved
        else:
            self._stats["total_cost_cloud"] += cost_usd

    def compute_local_cost(
        self, tokens: int, latency_ms: float
    ) -> float:
        """Compute electricity cost for a local inference request.

        Formula: (tokens/1000) * (gpu_watts/1000) * $/kWh * (latency_ms/3_600_000)
        """
        routing = self.config.routing
        cost = (
            (tokens / 1000)
            * (routing.gpu_tdp_watts / 1000)
            * routing.electricity_kwh_price
            * (latency_ms / 3_600_000)
        )
        return cost

    def get_routing_log(self) -> list[dict]:
        """Return the last 500 routing decisions."""
        return list(self._routing_log)

    def get_stats(self) -> dict:
        """Return aggregate statistics."""
        total = self._stats["total_requests"]
        sc = self._stats["smart_classifications"]
        return {
            **self._stats,
            "local_pct": (
                round(self._stats["local_requests"] / total * 100, 1)
                if total > 0 else 0
            ),
            "cloud_pct": (
                round(self._stats["cloud_requests"] / total * 100, 1)
                if total > 0 else 0
            ),
            "avg_classify_ms": (
                round(self._stats["total_classify_ms"] / sc, 1)
                if sc > 0 else 0.0
            ),
            "total_cost_saved": round(self._stats["total_saved"], 6),
        }

    def get_provider_statuses(
        self,
        provider_names: list[str],
        provider_types: dict[str, str],
    ) -> list[dict]:
        """Return health and circuit breaker status for all providers."""
        statuses = []
        for name in provider_names:
            circuit = self._get_circuit(name)
            statuses.append({
                "name": name,
                "type": provider_types.get(name, "unknown"),
                "healthy": not circuit.failed,
                "consecutive_errors": circuit.consecutive_errors,
                "circuit_open": circuit.failed,
                "failed_at": (
                    circuit.failed_at.isoformat()
                    if circuit.failed_at else None
                ),
            })
        return statuses

    def record_latency(self, provider_name: str, latency_ms: float) -> None:
        if provider_name not in self._latencies:
            self._latencies[provider_name] = deque(maxlen=500)
        self._latencies[provider_name].append(latency_ms)

    def get_latency_stats(self) -> dict:
        stats = {}
        for name, lat_list in self._latencies.items():
            if not lat_list:
                continue
            s_list = sorted(list(lat_list))
            stats[name] = {
                "p50": s_list[int(len(s_list)*0.5)],
                "p95": s_list[int(len(s_list)*0.95)],
                "p99": s_list[int(len(s_list)*0.99)],
                "samples": len(s_list),
            }
        return stats

    def set_mode(self, mode: str) -> None:
        """Change routing mode at runtime."""
        self.mode = PolicyMode(mode)
        logger.info(f"Routing mode changed to: {self.mode.value}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_circuit(self, name: str) -> CircuitState:
        """Get or create circuit breaker state for a provider."""
        if name not in self._circuits:
            self._circuits[name] = CircuitState()
        return self._circuits[name]

    def set_providers(self, providers: dict) -> None:
        """Register the provider objects for model-support scoring.

        Call this after providers are instantiated so the engine can
        check SUPPORTED_MODELS during routing decisions.
        """
        self._providers = providers

    def _score_providers(
        self,
        active: list[str],
        provider_types: dict[str, str],
        provider_costs: dict[str, float],
        vram_util: float,
        requested_model: str = "",
    ) -> list[ProviderScore]:
        """Score active providers based on the current policy mode."""
        scored = []
        provider_objects = getattr(self, "_providers", {})

        for name in active:
            ptype = provider_types.get(name, "cloud")
            cost = provider_costs.get(name, 0.0)
            is_local = ptype == "local"

            p95 = 500.0
            if name in self._latencies and len(self._latencies[name]) >= 10:
                s_list = sorted(list(self._latencies[name]))
                p95 = s_list[int(len(s_list)*0.95)]
            else:
                p95 = 200.0 if is_local else 400.0
            
            latency_penalty = p95 / 1000.0

            if self.mode == PolicyMode.COST:
                score = self._score_cost_mode(is_local, cost, vram_util) - latency_penalty * 0.2
            elif self.mode == PolicyMode.SPEED:
                score = self._score_speed_mode(is_local) - latency_penalty * 2.0
            elif self.mode == PolicyMode.FALLBACK:
                score = self._score_fallback_mode(
                    name, is_local, cost, vram_util
                ) - latency_penalty * 0.2
            else:
                score = self._score_cost_mode(is_local, cost, vram_util) - latency_penalty * 0.2

            # Model compatibility check: penalise providers that explicitly
            # declare SUPPORTED_MODELS and don't include the requested model.
            # Score of -2.0 ensures they are never chosen (circuit-broken = -1).
            if requested_model:
                prov = provider_objects.get(name)
                if prov is not None:
                    supported = getattr(prov, "SUPPORTED_MODELS", [])
                    if supported and requested_model not in supported:
                        score = -2.0

            scored.append(ProviderScore(
                name=name,
                provider_type=ptype,
                score=score,
                cost_per_1k=cost,
            ))

        return scored

    def _score_cost_mode(
        self, is_local: bool, cost: float, vram_util: float
    ) -> float:
        """Cost mode: local if VRAM < threshold, cheapest cloud otherwise."""
        threshold = 0.80
        if self.config.local_nodes:
            threshold = self.config.local_nodes[0].vram_threshold

        if is_local:
            if vram_util < threshold:
                return 1.0  # Local wins when VRAM is available
            else:
                return -0.5  # VRAM full — penalize local
        else:
            if vram_util >= threshold:
                # Cloud providers compete on cost (lower cost = higher score)
                return max(0.01, 0.8 - cost * 1000)
            else:
                # VRAM available — cloud is backup only
                return max(0.01, 0.3 - cost * 1000)

    def _score_speed_mode(self, is_local: bool) -> float:
        """Speed mode: always local. Cloud gets -1 (never unless all local fail)."""
        return 1.0 if is_local else -1.0

    def _score_fallback_mode(
        self,
        name: str,
        is_local: bool,
        cost: float,
        vram_util: float,
    ) -> float:
        """Fallback mode: local first, cloud on OOM/error or VRAM > 95%.

        Temporarily routes to cloud until VRAM recovers below 85%.
        """
        # Check if we should be spilling
        if vram_util > FALLBACK_VRAM_SPILL_THRESHOLD:
            self._fallback_spilling = True
        elif vram_util < FALLBACK_VRAM_RECOVER_THRESHOLD:
            self._fallback_spilling = False

        # Check if this local provider recently errored
        recently_errored = False
        if name in self._last_error:
            age = (datetime.utcnow() - self._last_error[name]).total_seconds()
            if age < 30:  # Consider errors within last 30s
                recently_errored = True

        if is_local:
            if self._fallback_spilling or recently_errored:
                return -0.5  # Temporarily avoid local
            return 1.0  # Local is preferred
        else:
            if self._fallback_spilling:
                # Cloud providers compete on cost during spill
                return max(0.01, 0.8 - cost * 1000)
            return 0.2  # Cloud is fallback

    def _explain_decision(
        self,
        winner: Optional[ProviderScore],
        vram_util: float,
    ) -> str:
        """Generate a human-readable reason for the routing decision."""
        if not winner:
            return "no_providers"

        if self.mode == PolicyMode.SPEED:
            return "speed_mode"

        if self.mode == PolicyMode.FALLBACK:
            if self._fallback_spilling:
                return "fallback_vram_spill"
            if winner.provider_type == "cloud":
                return "fallback_error"
            return "fallback_local_ok"

        # Cost mode
        if winner.provider_type == "local":
            return "vram_ok"
        return "vram_full"


# ── Legacy compatibility ─────────────────────────────────────────────────────
# Keep the old route_request function working for existing tests

def get_vram_snapshot(gpu_id: int = 0) -> VramSnapshot:
    """Get current GPU VRAM usage snapshot (legacy, use VramPoller instead).

    Args:
        gpu_id: GPU device ID (default: 0).

    Returns:
        VramSnapshot with current VRAM usage.

    Raises:
        RuntimeError: If pynvml not available or GPU not found.
    """
    try:
        import pynvml
    except ImportError:
        raise RuntimeError("pynvml not available. Install with: pip install pynvml")

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()

        return VramSnapshot(
            gpu_id=gpu_id,
            vram_used_gb=info.used / (1024**3),
            vram_free_gb=info.free / (1024**3),
            timestamp=datetime.now(),
        )
    except Exception as e:
        raise RuntimeError(f"Failed to query GPU {gpu_id}: {e}")


def route_request(vram: VramSnapshot, config: Config) -> RouteDecision:
    """Route request based on VRAM availability (legacy interface).

    Strategy:
    - If vram_free > threshold: route to local Ollama
    - Otherwise: route to cloud provider (Groq)
    """
    if not config.local_nodes and not config.cloud_providers:
        raise ValueError("No backends configured")

    threshold = 4.0
    if config.local_nodes:
        node = config.local_nodes[0]
        threshold = node.vram_threshold_gb or (node.vram_threshold * 10)

    if vram.vram_free_gb > threshold and config.local_nodes:
        backend = config.local_nodes[0].name
    elif config.cloud_providers:
        backend = config.cloud_providers[0].name
    else:
        raise ValueError("No suitable backend available")

    return RouteDecision(
        backend_chosen=backend,
        vram_at_decision=vram,
        latency_ms=0.0,
    )
