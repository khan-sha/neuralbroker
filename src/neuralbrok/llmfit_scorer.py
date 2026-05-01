"""
LLMFit-inspired composite model scoring engine.

Ports the core algorithm from llmfit-core (Rust) into Python.
Scores models across 4 dimensions — Quality, Speed, Fit, Context —
producing a composite score that determines the best model for
the user's hardware and workload.

Reference: https://github.com/AlexsJones/llmfit
"""
import logging
import math
import subprocess
from dataclasses import dataclass, field
from typing import Optional

from neuralbrok.quantization import (
    FitLevel,
    RunMode,
    QuantEstimate,
    full_estimate,
    best_quant_for_vram,
    estimate_weight_gb,
    QUANT_BPW,
)
from neuralbrok.models import ModelProfile, FALLBACK_MODELS

logger = logging.getLogger(__name__)


# ── Score Components ─────────────────────────────────────────────────────────

@dataclass
class ScoreComponents:
    """Multi-dimensional model score (0–100 each)."""
    quality: float = 0.0   # Model capability / intelligence
    speed: float = 0.0     # Estimated tok/s relative to interactive threshold
    fit: float = 0.0       # How well model fits in VRAM
    context: float = 0.0   # Context window size score
    composite: float = 0.0 # Weighted combination


@dataclass
class ModelFit:
    """Complete model evaluation result — the llmfit output format."""
    name: str
    family: str
    params_b: float
    best_quant: str
    run_mode: RunMode
    fit_level: FitLevel
    scores: ScoreComponents
    estimated_tok_s: float
    vram_needed_gb: float
    vram_available_gb: float
    headroom_gb: float
    capabilities: list[str] = field(default_factory=list)
    use_case_match: float = 0.0  # 0–1 how well model matches requested use case
    is_installed: bool = False
    ollama_tag: str = ""
    run_cmd: str = ""


@dataclass
class ScoringWeights:
    """Weights for composite score calculation (sum to 1.0)."""
    quality: float = 0.35
    speed: float = 0.25
    fit: float = 0.25
    context: float = 0.15


@dataclass
class SystemSpecs:
    """Hardware specification for scoring — mirrors llmfit's SystemSpecs."""
    cpu_cores: int = 4
    ram_gb: float = 16.0
    gpu_name: str = "Unknown"
    gpu_vendor: str = "none"
    vram_gb: float = 0.0
    bandwidth_gbps: float = 40.0
    platform: str = "unknown"
    # Detected local runtimes
    ollama_available: bool = False
    llamacpp_available: bool = False
    lmstudio_available: bool = False
    docker_model_runner: bool = False


# ── Use-Case Definitions ─────────────────────────────────────────────────────
# Maps use-case names to preferred capabilities (with weights)

USE_CASE_CAPS: dict[str, dict[str, float]] = {
    "coding":      {"coding": 1.0, "code": 0.8, "agentic": 0.4, "tools": 0.3},
    "reasoning":   {"reasoning": 1.0, "math": 0.7, "agentic": 0.3},
    "math":        {"math": 1.0, "reasoning": 0.6},
    "chat":        {"chat": 1.0, "tools": 0.3, "agentic": 0.3},
    "vision":      {"vision": 1.0},
    "tools":       {"tools": 1.0, "agentic": 0.8, "coding": 0.4},
    "agentic":     {"agentic": 1.0, "tools": 0.7, "coding": 0.4, "reasoning": 0.3},
    "long_context": {"long_context": 1.0, "reasoning": 0.3},
    "rag":         {"long_context": 0.8, "reasoning": 0.5, "chat": 0.3},
    "general":     {"chat": 0.5, "coding": 0.3, "reasoning": 0.2},
}

# Intelligence anchors — AA Intelligence Index scores for known model families
FAMILY_INTELLIGENCE: dict[str, float] = {
    "deepseek-r1":      27.0,
    "qwq":              30.0,
    "qwen3":            30.0,
    "qwen3.5":          32.0,
    "qwen2.5-coder":    22.0,
    "qwen2.5":          18.0,
    "gemma-4":          22.0,
    "gemma-3":          16.0,
    "llama-4":          18.0,
    "llama3.3":         14.0,
    "mistral":          15.0,
    "phi-4":            18.0,
    "gpt-oss":          25.0,
    "glm":              20.0,
    "nemotron":         22.0,
    "codestral":        20.0,
    "devstral":         18.0,
    "magistral":        22.0,
}


# ── Scoring Functions ────────────────────────────────────────────────────────

def _score_quality(model: ModelProfile) -> float:
    """Score 0–100 based on model intelligence/capability.

    Uses intelligence_score from registry, family lookup, or params-based heuristic.
    """
    # Explicit intelligence score (best source)
    if model.intelligence_score > 0:
        # Scale 0–60 AA index → 0–100
        return min(100.0, model.intelligence_score * 1.67)

    # Family lookup
    name_lc = model.name.lower()
    for family, score in FAMILY_INTELLIGENCE.items():
        if family in name_lc:
            # Scale by log2(params) — bigger models in same family score higher
            scale = 1.0 + 0.1 * math.log2(max(model.params_b, 1.0))
            return min(100.0, score * scale * 1.67)

    # Pure params heuristic (last resort)
    # 1B→15, 7B→35, 14B→50, 32B→65, 70B→80
    return min(100.0, math.log2(max(model.params_b, 0.5)) * 10.0 + 5.0)


def _score_speed(
    model: ModelProfile,
    hw: SystemSpecs,
    quant_estimate: QuantEstimate,
) -> float:
    """Score 0–100 based on estimated tokens/second.

    Uses bandwidth-based estimation: tok/s ≈ bandwidth / weight.
    Interactive threshold: 20 tok/s = score 50.
    """
    weight = quant_estimate.weight_gb
    if weight <= 0:
        weight = estimate_weight_gb(model.params_b, quant_estimate.quant)

    bandwidth = hw.bandwidth_gbps or 40.0

    if quant_estimate.run_mode == RunMode.CPU_ONLY:
        # CPU inference is ~1-5 tok/s for most models
        est_tok_s = max(0.5, 8.0 / max(model.params_b, 1.0) * (hw.cpu_cores / 4.0))
    elif quant_estimate.run_mode == RunMode.GPU_OFFLOAD:
        # Partial offload — estimate based on fraction on GPU
        gpu_frac = quant_estimate.gpu_layers / max(quant_estimate.total_layers, 1)
        full_gpu_tok_s = bandwidth / max(weight + 0.5, 0.1)
        est_tok_s = full_gpu_tok_s * (0.3 + 0.7 * gpu_frac)
    else:
        # Full GPU — bandwidth / weight
        est_tok_s = bandwidth / max(weight + 0.5, 0.1)

    # Map tok/s to 0–100 score
    # 0 tok/s → 0, 20 tok/s → 50, 50 tok/s → 75, 100+ tok/s → 95
    if est_tok_s <= 0:
        return 0.0
    return min(100.0, 50.0 * math.log2(est_tok_s / 20.0 + 1.0) + 50.0)


def _estimated_tok_s(
    model: ModelProfile,
    hw: SystemSpecs,
    quant_estimate: QuantEstimate,
) -> float:
    """Return raw estimated tok/s (not scored)."""
    weight = quant_estimate.weight_gb
    if weight <= 0:
        weight = estimate_weight_gb(model.params_b, quant_estimate.quant)

    bandwidth = hw.bandwidth_gbps or 40.0

    if quant_estimate.run_mode == RunMode.CPU_ONLY:
        return max(0.5, 8.0 / max(model.params_b, 1.0) * (hw.cpu_cores / 4.0))
    elif quant_estimate.run_mode == RunMode.GPU_OFFLOAD:
        gpu_frac = quant_estimate.gpu_layers / max(quant_estimate.total_layers, 1)
        full = bandwidth / max(weight + 0.5, 0.1)
        return full * (0.3 + 0.7 * gpu_frac)
    else:
        return bandwidth / max(weight + 0.5, 0.1)


def _score_fit(quant_estimate: QuantEstimate) -> float:
    """Score 0–100 based on how well model fits in VRAM.

    Comfortable → 90–100, Tight → 60–89, Partial → 20–59, TooLarge → 0.
    """
    if quant_estimate.fit_level == FitLevel.COMFORTABLE:
        # Scale by headroom: 2GB → 90, 4GB+ → 100
        return min(100.0, 90.0 + quant_estimate.headroom_gb * 2.5)
    elif quant_estimate.fit_level == FitLevel.TIGHT:
        # Scale by headroom: 0.5GB → 60, 2GB → 89
        return 60.0 + max(0, quant_estimate.headroom_gb - 0.5) * 19.3
    elif quant_estimate.fit_level == FitLevel.PARTIAL:
        # GPU offload — score by GPU layer fraction
        frac = quant_estimate.gpu_layers / max(quant_estimate.total_layers, 1)
        return 20.0 + frac * 39.0
    else:
        return 0.0


def _score_context(model: ModelProfile) -> float:
    """Score 0–100 based on context window size.

    4k → 20, 8k → 35, 32k → 55, 128k → 75, 262k → 88, 1M → 100.
    """
    ctx_k = model.ctx_k if model.ctx_k > 0 else 4
    if ctx_k >= 1000:
        return 100.0
    elif ctx_k >= 262:
        return 88.0
    elif ctx_k >= 128:
        return 75.0
    elif ctx_k >= 32:
        return 55.0
    elif ctx_k >= 8:
        return 35.0
    else:
        return 20.0


def _use_case_match(model: ModelProfile, use_case: str) -> float:
    """Return 0–1 match score between model capabilities and use case."""
    caps = USE_CASE_CAPS.get(use_case, USE_CASE_CAPS["general"])
    if not caps:
        return 0.5

    total_weight = sum(caps.values())
    matched = 0.0
    for cap, weight in caps.items():
        if cap in model.capabilities:
            matched += weight
    return matched / total_weight if total_weight > 0 else 0.5


# ── Main Scoring API ─────────────────────────────────────────────────────────

def score_model(
    model: ModelProfile,
    hw: SystemSpecs,
    weights: Optional[ScoringWeights] = None,
    use_case: str = "general",
    context_k: int = 4,
) -> ModelFit:
    """Score a single model against hardware specs.

    Returns a ModelFit with multi-dimensional scores and metadata.
    This is the core llmfit algorithm ported to Python.
    """
    if weights is None:
        weights = ScoringWeights()

    # Step 1: Find best quantization and estimate VRAM
    qe = full_estimate(
        params_b=model.params_b,
        available_vram_gb=hw.vram_gb,
        available_ram_gb=hw.ram_gb,
        context_k=context_k,
        n_layers=model.layers,
        total_layers=model.layers,
    )

    # Step 2: Score each dimension
    quality = _score_quality(model)
    speed = _score_speed(model, hw, qe)
    fit = _score_fit(qe)
    context = _score_context(model)

    # Step 3: Use-case adjustment — boost quality score for matching capabilities
    uc_match = _use_case_match(model, use_case)
    quality_adjusted = quality * (0.7 + 0.3 * uc_match)

    # Step 4: Composite score (weighted sum)
    composite = (
        quality_adjusted * weights.quality +
        speed * weights.speed +
        fit * weights.fit +
        context * weights.context
    )

    # Step 5: Penalty for models that won't run at all
    if qe.fit_level == FitLevel.TOO_LARGE:
        composite *= 0.1  # Massive penalty but don't zero out (show in UI)

    scores = ScoreComponents(
        quality=round(quality_adjusted, 1),
        speed=round(speed, 1),
        fit=round(fit, 1),
        context=round(context, 1),
        composite=round(composite, 1),
    )

    est_tok_s = _estimated_tok_s(model, hw, qe)

    return ModelFit(
        name=model.name,
        family=model.family,
        params_b=model.params_b,
        best_quant=qe.quant,
        run_mode=qe.run_mode,
        fit_level=qe.fit_level,
        scores=scores,
        estimated_tok_s=round(est_tok_s, 1),
        vram_needed_gb=qe.total_vram_gb,
        vram_available_gb=hw.vram_gb,
        headroom_gb=qe.headroom_gb,
        capabilities=model.capabilities,
        use_case_match=round(uc_match, 2),
        is_installed=model.is_installed,
        ollama_tag=model.ollama_tag,
        run_cmd=f"ollama run {model.ollama_tag}",
    )


def rank_models(
    hw: SystemSpecs,
    use_case: str = "general",
    weights: Optional[ScoringWeights] = None,
    models: Optional[list[ModelProfile]] = None,
    max_results: int = 20,
    include_too_large: bool = False,
    context_k: int = 4,
) -> list[ModelFit]:
    """Rank all models against hardware specs, sorted by composite score.

    This is the main entry point — equivalent to `llmfit recommend`.
    """
    if models is None:
        models = FALLBACK_MODELS
    if weights is None:
        weights = ScoringWeights()

    results: list[ModelFit] = []
    for model in models:
        fit = score_model(model, hw, weights, use_case, context_k)
        if not include_too_large and fit.fit_level == FitLevel.TOO_LARGE:
            continue
        results.append(fit)

    # Sort by composite score descending
    results.sort(key=lambda f: f.scores.composite, reverse=True)
    return results[:max_results]


def detect_system_specs() -> SystemSpecs:
    """Detect hardware specs using NeuralBroker's existing detection.

    Wraps detect_device() and adds provider availability checks.
    """
    from neuralbrok.detect import detect_device

    profile = detect_device()

    specs = SystemSpecs(
        cpu_cores=_get_cpu_cores(),
        ram_gb=profile.ram_gb,
        gpu_name=profile.gpu_model,
        gpu_vendor=profile.gpu_vendor,
        vram_gb=profile.vram_gb,
        bandwidth_gbps=profile.bandwidth_gbps or 40.0,
        platform=profile.platform,
    )

    # Check local runtime availability
    specs.ollama_available = _check_runtime("http://localhost:11434/api/tags")
    specs.llamacpp_available = _check_runtime("http://localhost:8080/health")
    specs.lmstudio_available = _check_runtime("http://localhost:1234/v1/models")
    specs.docker_model_runner = _check_runtime("http://localhost:12434/engines/llama.cpp/v1/models")

    return specs


def _get_cpu_cores() -> int:
    """Get CPU core count."""
    try:
        import os
        return os.cpu_count() or 4
    except Exception:
        return 4


def _check_runtime(url: str) -> bool:
    """Check if a local runtime is available (synchronous, fast timeout)."""
    try:
        import urllib.request
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def model_fit_to_dict(fit: ModelFit) -> dict:
    """Convert ModelFit to JSON-serializable dict for API responses."""
    return {
        "name": fit.name,
        "family": fit.family,
        "params_b": fit.params_b,
        "best_quant": fit.best_quant,
        "run_mode": fit.run_mode.value,
        "fit_level": fit.fit_level.value,
        "scores": {
            "quality": fit.scores.quality,
            "speed": fit.scores.speed,
            "fit": fit.scores.fit,
            "context": fit.scores.context,
            "composite": fit.scores.composite,
        },
        "estimated_tok_s": fit.estimated_tok_s,
        "vram_needed_gb": fit.vram_needed_gb,
        "vram_available_gb": fit.vram_available_gb,
        "headroom_gb": fit.headroom_gb,
        "capabilities": fit.capabilities,
        "use_case_match": fit.use_case_match,
        "is_installed": fit.is_installed,
        "ollama_tag": fit.ollama_tag,
        "run_cmd": fit.run_cmd,
    }
