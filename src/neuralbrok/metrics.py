"""
Prometheus metrics for NeuralBroker.

Exposes GET /metrics in Prometheus text format.
Uses the prometheus_client library — only new dependency added.

If prometheus_client is not installed, all metric operations are no-ops
so the rest of the system works without it.
"""
import logging
import time

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logger.warning("prometheus_client not installed — metrics disabled")


# ── Metric Definitions ───────────────────────────────────────────────────────

if HAS_PROMETHEUS:
    REQUESTS_TOTAL = Counter(
        "nb_requests_total",
        "Total requests routed by NeuralBroker",
        ["backend", "mode", "status"],
    )

    ROUTING_LATENCY = Histogram(
        "nb_routing_latency_ms",
        "Routing decision latency in milliseconds",
        buckets=[0.1, 0.5, 1, 2, 5, 10, 25, 50],
    )

    VRAM_UTILIZATION = Gauge(
        "nb_vram_utilization",
        "GPU VRAM utilization ratio (0.0 - 1.0)",
        ["gpu_index"],
    )

    COST_PER_REQUEST = Histogram(
        "nb_cost_per_request",
        "Cost in USD per routed request",
        ["backend"],
        buckets=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
    )

    PROVIDER_ERRORS = Counter(
        "nb_provider_errors_total",
        "Total errors by provider",
        ["provider"],
    )

    TOKENS_LOCAL = Counter(
        "nb_tokens_local_total",
        "Total tokens processed by local providers",
    )

    TOKENS_CLOUD = Counter(
        "nb_tokens_cloud_total",
        "Total tokens processed by cloud providers",
    )


# ── Public API ────────────────────────────────────────────────────────────────

def record_request(backend: str, mode: str, status: str) -> None:
    """Record a routed request."""
    if HAS_PROMETHEUS:
        REQUESTS_TOTAL.labels(backend=backend, mode=mode, status=status).inc()


def record_routing_latency(latency_ms: float) -> None:
    """Record routing decision latency."""
    if HAS_PROMETHEUS:
        ROUTING_LATENCY.observe(latency_ms)


def set_vram_utilization(gpu_index: int, utilization: float) -> None:
    """Set current VRAM utilization gauge."""
    if HAS_PROMETHEUS:
        VRAM_UTILIZATION.labels(gpu_index=str(gpu_index)).set(utilization)


def record_cost(backend: str, cost_usd: float) -> None:
    """Record cost for a single request."""
    if HAS_PROMETHEUS:
        COST_PER_REQUEST.labels(backend=backend).observe(cost_usd)


def record_provider_error(provider: str) -> None:
    """Record a provider error."""
    if HAS_PROMETHEUS:
        PROVIDER_ERRORS.labels(provider=provider).inc()


def record_tokens(local: bool, count: int) -> None:
    """Record token count for local or cloud."""
    if HAS_PROMETHEUS:
        if local:
            TOKENS_LOCAL.inc(count)
        else:
            TOKENS_CLOUD.inc(count)


def get_metrics_response() -> tuple[str, str]:
    """Generate Prometheus metrics text and content type.

    Returns:
        Tuple of (metrics_text, content_type).
    """
    if HAS_PROMETHEUS:
        return generate_latest().decode("utf-8"), CONTENT_TYPE_LATEST
    return "# prometheus_client not installed\n", "text/plain"
