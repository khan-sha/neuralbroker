"""
Quantization-aware VRAM estimation engine.

Ported from neuralfit-core's quantization logic — maps quantization levels
to bits-per-weight and calculates precise memory requirements for model
weights and KV cache at any context length.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ── Quantization Levels ──────────────────────────────────────────────────────
# Bits-per-weight for GGUF quantization formats.
# Source: neuralfit-core/src/fit.rs + ggml quantization spec

QUANT_BPW: dict[str, float] = {
    "Q2_K":   2.63,
    "Q3_K_S": 3.44,
    "Q3_K_M": 3.91,
    "Q3_K_L": 4.27,
    "Q4_0":   4.50,
    "Q4_K_S": 4.58,
    "Q4_K_M": 4.87,
    "Q4_1":   5.00,
    "Q5_0":   5.50,
    "Q5_K_S": 5.54,
    "Q5_K_M": 5.69,
    "Q5_1":   6.00,
    "Q6_K":   6.56,
    "Q8_0":   8.50,
    "FP16":  16.00,
    "FP32":  32.00,
}

# Preferred quantization order (best quality → smallest size)
QUANT_PREFERENCE = [
    "FP16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0",
    "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q2_K",
]

# Default quant for Ollama downloads
DEFAULT_QUANT = "Q4_K_M"


class FitLevel(str, Enum):
    """How well a model fits available VRAM."""
    COMFORTABLE = "comfortable"   # > 2 GB headroom
    TIGHT       = "tight"         # 0.5–2 GB headroom
    PARTIAL     = "partial"       # Needs CPU offload
    TOO_LARGE   = "too_large"     # Won't run at all


class RunMode(str, Enum):
    """How the model will run."""
    FULL_GPU    = "full_gpu"       # Entire model in VRAM
    GPU_OFFLOAD = "gpu_offload"    # Split GPU/CPU
    CPU_ONLY    = "cpu_only"       # No GPU, pure CPU inference


@dataclass
class QuantEstimate:
    """VRAM estimate for a specific model + quantization combo."""
    quant: str
    bits_per_weight: float
    weight_gb: float
    kv_cache_gb: float        # At default context (4k)
    total_vram_gb: float      # weight + kv_cache + overhead
    fit_level: FitLevel
    run_mode: RunMode
    headroom_gb: float        # Available VRAM - total needed
    gpu_layers: int           # Layers on GPU (for partial offload)
    total_layers: int         # Total model layers


# ── Core Functions ───────────────────────────────────────────────────────────

def estimate_weight_gb(params_b: float, quant: str = DEFAULT_QUANT) -> float:
    """Calculate model weight size in GB for a given quantization level.

    Formula: params_billions × bits_per_weight / 8
    Plus ~5% overhead for GGUF metadata/tensors.
    """
    bpw = QUANT_BPW.get(quant, QUANT_BPW[DEFAULT_QUANT])
    raw_gb = params_b * bpw / 8.0
    return raw_gb * 1.05  # 5% overhead


def estimate_kv_cache_gb(
    params_b: float,
    context_k: int = 4,
    n_layers: int = 0,
    n_heads: int = 0,
    head_dim: int = 128,
) -> float:
    """Estimate KV cache size at a given context length.

    If layer/head info isn't available, use heuristic:
    ~0.5 GB per 1k context per 10B params (Q4 KV).
    """
    if n_layers > 0 and n_heads > 0:
        # Precise: 2 (K+V) × layers × heads × head_dim × context × 2 bytes (FP16)
        kv_bytes = 2 * n_layers * n_heads * head_dim * (context_k * 1024) * 2
        return kv_bytes / (1024**3)

    # Heuristic fallback — scales with params and context
    return (params_b / 10.0) * (context_k / 4.0) * 0.5


def estimate_total_vram(
    params_b: float,
    quant: str = DEFAULT_QUANT,
    context_k: int = 4,
    n_layers: int = 0,
    n_heads: int = 0,
) -> float:
    """Total VRAM: weights + KV cache + runtime overhead (~300 MB)."""
    weight = estimate_weight_gb(params_b, quant)
    kv = estimate_kv_cache_gb(params_b, context_k, n_layers, n_heads)
    overhead = 0.3  # CUDA context, activations, etc.
    return weight + kv + overhead


def best_quant_for_vram(
    params_b: float,
    available_vram_gb: float,
    context_k: int = 4,
    n_layers: int = 0,
    min_headroom_gb: float = 0.5,
) -> Optional[str]:
    """Find the highest-quality quantization that fits in available VRAM.

    Iterates from FP16 down to Q2_K, returning the first that fits
    with at least `min_headroom_gb` of spare VRAM.
    """
    for quant in QUANT_PREFERENCE:
        total = estimate_total_vram(params_b, quant, context_k, n_layers)
        if total + min_headroom_gb <= available_vram_gb:
            return quant
    return None  # Nothing fits


def full_estimate(
    params_b: float,
    available_vram_gb: float,
    available_ram_gb: float = 32.0,
    quant: Optional[str] = None,
    context_k: int = 4,
    n_layers: int = 0,
    total_layers: int = 0,
) -> QuantEstimate:
    """Full VRAM analysis for a model — determines fit level, run mode, and optimal quant.

    Returns a QuantEstimate with everything needed for the neuralfit-style scoring.
    """
    # Auto-select best quant if not specified
    if quant is None:
        quant = best_quant_for_vram(params_b, available_vram_gb, context_k, n_layers) or "Q4_K_M"

    bpw = QUANT_BPW.get(quant, QUANT_BPW[DEFAULT_QUANT])
    weight = estimate_weight_gb(params_b, quant)
    kv = estimate_kv_cache_gb(params_b, context_k, n_layers)
    overhead = 0.3
    total = weight + kv + overhead
    headroom = available_vram_gb - total

    # Determine fit level
    if headroom >= 2.0:
        fit_level = FitLevel.COMFORTABLE
    elif headroom >= 0.5:
        fit_level = FitLevel.TIGHT
    elif headroom >= -available_ram_gb * 0.5:
        fit_level = FitLevel.PARTIAL
    else:
        fit_level = FitLevel.TOO_LARGE

    # Determine run mode
    if headroom >= 0:
        run_mode = RunMode.FULL_GPU
    elif available_vram_gb > 2.0:
        run_mode = RunMode.GPU_OFFLOAD
    else:
        run_mode = RunMode.CPU_ONLY

    # Estimate GPU layers for partial offload
    if total_layers == 0:
        # Heuristic: ~2 layers per billion params for transformer models
        total_layers = max(int(params_b * 2), 4)

    if run_mode == RunMode.FULL_GPU:
        gpu_layers = total_layers
    elif run_mode == RunMode.GPU_OFFLOAD:
        # Proportion of layers that fit in VRAM
        frac = max(0.0, available_vram_gb / max(total, 0.1))
        gpu_layers = min(total_layers, max(1, int(total_layers * frac)))
    else:
        gpu_layers = 0

    return QuantEstimate(
        quant=quant,
        bits_per_weight=bpw,
        weight_gb=round(weight, 2),
        kv_cache_gb=round(kv, 2),
        total_vram_gb=round(total, 2),
        fit_level=fit_level,
        run_mode=run_mode,
        headroom_gb=round(headroom, 2),
        gpu_layers=gpu_layers,
        total_layers=total_layers,
    )
