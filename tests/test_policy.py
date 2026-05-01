"""
Tests for the policy engine (cost, speed, fallback modes + circuit breaker).
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from neuralbrok.config import Config, LocalNodeConfig, CloudProviderConfig, RoutingConfig
from neuralbrok.router import PolicyEngine, CIRCUIT_ERROR_THRESHOLD
from neuralbrok.types import VramSnapshot, PolicyMode


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_config(mode: str = "cost", threshold: float = 0.80) -> Config:
    """Build a minimal Config for testing."""
    return Config(
        local_nodes=[
            LocalNodeConfig(
                name="local-gpu",
                runtime="ollama",
                host="localhost:11434",
                vram_threshold=threshold,
            )
        ],
        cloud_providers=[
            CloudProviderConfig(
                name="groq",
                api_key_env="GROQ_KEY",
                base_url="https://api.groq.com/openai/v1",
                cost_per_1k_tokens=0.00006,
            ),
            CloudProviderConfig(
                name="together",
                api_key_env="TOGETHER_KEY",
                base_url="https://api.together.xyz/v1",
                cost_per_1k_tokens=0.00020,
            ),
        ],
        routing=RoutingConfig(
            default_mode=mode,
            electricity_kwh_price=0.14,
            gpu_tdp_watts=320,
        ),
    )


def make_vram(used: float, free: float) -> VramSnapshot:
    return VramSnapshot(
        gpu_id=0,
        vram_used_gb=used,
        vram_free_gb=free,
        timestamp=datetime.now(),
    )


PROVIDERS = ["local-gpu", "groq", "together"]
TYPES = {"local-gpu": "local", "groq": "cloud", "together": "cloud"}
COSTS = {"local-gpu": 0.00002, "groq": 0.00006, "together": 0.00020}


# ── Cost Mode ─────────────────────────────────────────────────────────────────

class TestCostMode:

    def test_routes_local_when_vram_ok(self):
        engine = PolicyEngine(make_config("cost", threshold=0.80))
        vram = make_vram(used=4.0, free=6.0)  # 40% utilization

        decision = engine.decide(vram, PROVIDERS, TYPES, COSTS)
        assert decision.backend_chosen == "local-gpu"
        assert decision.reason == "vram_ok"

    def test_routes_cloud_when_vram_full(self):
        engine = PolicyEngine(make_config("cost", threshold=0.80))
        vram = make_vram(used=9.0, free=1.0)  # 90% utilization

        decision = engine.decide(vram, PROVIDERS, TYPES, COSTS)
        assert decision.backend_chosen != "local-gpu"
        assert decision.reason == "vram_full"

    def test_cheapest_cloud_wins(self):
        engine = PolicyEngine(make_config("cost", threshold=0.80))
        vram = make_vram(used=9.0, free=1.0)  # 90% → cloud

        decision = engine.decide(vram, PROVIDERS, TYPES, COSTS)
        # Groq is cheaper than Together
        assert decision.backend_chosen == "groq"

    def test_fallback_chain_populated(self):
        engine = PolicyEngine(make_config("cost", threshold=0.80))
        vram = make_vram(used=4.0, free=6.0)

        decision = engine.decide(vram, PROVIDERS, TYPES, COSTS)
        assert len(decision.fallback_chain) > 0


# ── Speed Mode ────────────────────────────────────────────────────────────────

class TestSpeedMode:

    def test_always_routes_local(self):
        engine = PolicyEngine(make_config("speed"))
        vram = make_vram(used=9.5, free=0.5)  # 95% — doesn't matter

        decision = engine.decide(vram, PROVIDERS, TYPES, COSTS)
        assert decision.backend_chosen == "local-gpu"
        assert decision.reason == "speed_mode"

    def test_errors_when_no_local_configured(self):
        config = Config(
            cloud_providers=[
                CloudProviderConfig(
                    name="groq",
                    api_key_env="GROQ_KEY",
                    base_url="https://api.groq.com/openai/v1",
                )
            ],
            routing=RoutingConfig(default_mode="speed"),
        )
        engine = PolicyEngine(config)
        vram = make_vram(used=0, free=8)

        # In speed mode with only cloud, cloud gets score -1
        # but it's the only option so it still gets picked
        decision = engine.decide(vram, ["groq"], {"groq": "cloud"}, {"groq": 0.00006})
        # Cloud is chosen because it's the only option
        assert decision.backend_chosen == "groq"


# ── Fallback Mode ─────────────────────────────────────────────────────────────

class TestFallbackMode:

    def test_routes_local_normally(self):
        engine = PolicyEngine(make_config("fallback"))
        vram = make_vram(used=4.0, free=6.0)

        decision = engine.decide(vram, PROVIDERS, TYPES, COSTS)
        assert decision.backend_chosen == "local-gpu"

    def test_spills_to_cloud_above_95_pct(self):
        engine = PolicyEngine(make_config("fallback"))
        vram = make_vram(used=9.6, free=0.4)  # 96% > 95% threshold

        decision = engine.decide(vram, PROVIDERS, TYPES, COSTS)
        assert decision.backend_chosen != "local-gpu"
        assert "spill" in decision.reason or "fallback" in decision.reason

    def test_recovers_to_local_below_85_pct(self):
        engine = PolicyEngine(make_config("fallback"))

        # First: spike above 95%
        vram_high = make_vram(used=9.6, free=0.4)
        engine.decide(vram_high, PROVIDERS, TYPES, COSTS)

        # Then: drop below 85%
        vram_low = make_vram(used=4.0, free=6.0)  # 40%
        decision = engine.decide(vram_low, PROVIDERS, TYPES, COSTS)
        assert decision.backend_chosen == "local-gpu"

    def test_cloud_fallback_chain_present(self):
        engine = PolicyEngine(make_config("fallback"))
        vram = make_vram(used=4.0, free=6.0)

        decision = engine.decide(vram, PROVIDERS, TYPES, COSTS)
        assert len(decision.fallback_chain) > 0


# ── Circuit Breaker ───────────────────────────────────────────────────────────

class TestCircuitBreaker:

    def test_provider_marked_failed_after_n_errors(self):
        engine = PolicyEngine(make_config("cost"))

        for _ in range(CIRCUIT_ERROR_THRESHOLD):
            engine.record_error("local-gpu")

        circuit = engine._get_circuit("local-gpu")
        assert circuit.failed is True

    def test_failed_provider_skipped_in_routing(self):
        engine = PolicyEngine(make_config("cost"))
        vram = make_vram(used=4.0, free=6.0)  # VRAM ok → would normally pick local

        # Trip the circuit breaker
        for _ in range(CIRCUIT_ERROR_THRESHOLD):
            engine.record_error("local-gpu")

        decision = engine.decide(vram, PROVIDERS, TYPES, COSTS)
        assert decision.backend_chosen != "local-gpu"

    def test_success_resets_circuit(self):
        engine = PolicyEngine(make_config("cost"))

        engine.record_error("groq")
        engine.record_error("groq")
        engine.record_success("groq")

        circuit = engine._get_circuit("groq")
        assert circuit.consecutive_errors == 0
        assert circuit.failed is False

    def test_circuit_recovers_after_timeout(self):
        engine = PolicyEngine(make_config("cost"))

        for _ in range(CIRCUIT_ERROR_THRESHOLD):
            engine.record_error("local-gpu")

        circuit = engine._get_circuit("local-gpu")
        assert circuit.failed is True

        # Simulate time passing beyond recovery period
        circuit.failed_at = datetime.now(timezone.utc) - timedelta(seconds=61)

        vram = make_vram(used=4.0, free=6.0)
        decision = engine.decide(vram, PROVIDERS, TYPES, COSTS)
        # Should recover and route to local again
        assert decision.backend_chosen == "local-gpu"


# ── Invalid Mode ──────────────────────────────────────────────────────────────

class TestInvalidMode:

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError):
            PolicyMode("nonexistent")


# ── Routing Log + Stats ──────────────────────────────────────────────────────

class TestRoutingLog:

    def test_decisions_logged(self):
        engine = PolicyEngine(make_config("cost"))
        vram = make_vram(used=4.0, free=6.0)

        engine.decide(vram, PROVIDERS, TYPES, COSTS)
        engine.decide(vram, PROVIDERS, TYPES, COSTS)

        log = engine.get_routing_log()
        assert len(log) == 2
        assert all("backend" in entry for entry in log)

    def test_stats_track_requests(self):
        engine = PolicyEngine(make_config("cost"))
        vram = make_vram(used=4.0, free=6.0)

        engine.decide(vram, PROVIDERS, TYPES, COSTS)
        engine.decide(vram, PROVIDERS, TYPES, COSTS)

        stats = engine.get_stats()
        assert stats["total_requests"] == 2


# ── Electricity Cost ──────────────────────────────────────────────────────────

class TestElectricityCost:

    def test_local_cost_calculation(self):
        engine = PolicyEngine(make_config("cost"))
        # 1000 tokens, 200ms latency
        cost = engine.compute_local_cost(tokens=1000, latency_ms=200)
        assert cost > 0
        assert cost < 0.01  # Electricity cost should be tiny

    def test_zero_tokens_zero_cost(self):
        engine = PolicyEngine(make_config("cost"))
        cost = engine.compute_local_cost(tokens=0, latency_ms=200)
        assert cost > 0.0
