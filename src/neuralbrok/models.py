from dataclasses import dataclass, field
import asyncio
import json
import time
import os
from pathlib import Path
from typing import Optional, List, Dict
import httpx

# ── 2026 Hardened Model Registry ──────────────────────────────────────────
# Maps workloads and legacy aliases to modern, high-performance models.
MODEL_REGISTRY: Dict[str, str] = {
    "default":         "qwen3:7b",
    "coding":          "qwen2.5-coder:7b",
    "coding_large":    "qwen3-coder:30b",
    "reasoning":       "deepseek-r1:7b",
    "reasoning_large": "deepseek-r1:32b",
    "vision":          "gemma4:9b",
    "vision_large":    "gemma4:26b",
    "long_context":    "llama4:8b",
    "small":           "phi-4-mini",
    "small_mid":       "phi-4",
    "cloud_default":   "gpt-oss:120b-cloud",
    "cloud_coding":    "qwen3-coder:480b-cloud",
    "llama2":          "qwen3:7b",
    "llama3":          "llama4:8b",
    "codellama":       "qwen2.5-coder:7b",
    "mistral":         "qwen3:7b",
    "gemma":           "gemma4:9b",
    "gemma2":          "gemma4:9b",
}

def resolve_model(name: str) -> str:
    """Resolve a model name or alias to its registry-recommended tag."""
    return MODEL_REGISTRY.get(name, name)

@dataclass
class ModelProfile:
    name: str
    family: str
    params_b: float
    quant: str
    vram_gb: float  # Legacy/Fallback
    ram_gb: float
    ctx_k: int
    tok_per_sec_gpu: dict
    tok_per_sec_cpu: float
    capabilities: list[str]
    recommended_for: list[str]
    ollama_tag: str
    notes: str
    weight_gb: float = 0.0  # Precision field from whatmodels
    kv_per_1k_gb: float = 0.0  # Precision field from whatmodels
    layers: int = 0
    is_installed: bool = False
    vram_estimated_gb: float = 0.0
    intelligence_score: float = 0.0  # Artificial Analysis Intelligence Index (0-60 scale)

def estimate_vram_requirement(model: ModelProfile, context_k: int) -> float:
    """Calculate precise VRAM requirement: weights + KV cache."""
    if model.weight_gb > 0 and model.kv_per_1k_gb > 0:
        return model.weight_gb + (model.kv_per_1k_gb * context_k)
    return model.vram_gb # Fallback

async def validate_models(host: str = "localhost:11434") -> Dict[str, bool]:
    """Check which models from the registry are installed locally."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"http://{host}/api/tags")
            if resp.status_code != 200:
                return {}
            installed = {m["name"] for m in resp.json().get("models", [])}
            
            results = {}
            for alias, tag in MODEL_REGISTRY.items():
                if not tag.endswith(":cloud"):
                    results[tag] = tag in installed
            return results
    except Exception:
        return {}

MODELS = [
    {"tag": "llama3.2:1b", "params_b": 1.24, "vram_gb": 0.9, "capabilities": ['tools', 'chat', 'agentic']},
    {"tag": "llama3.2:3b", "params_b": 3.21, "vram_gb": 6.9, "capabilities": ['tools', 'chat', 'agentic']},
    {"tag": "llama3.1:8b", "params_b": 8.03, "vram_gb": 16.5, "capabilities": ['chat', 'agentic', 'code', 'coding', 'tools']},
    {"tag": "llama3.2-11b:vision", "params_b": 10.6, "vram_gb": 6.4, "capabilities": ['tools', 'vision', 'chat', 'agentic']},
    {"tag": "llama3.3:70b", "params_b": 70.6, "vram_gb": 43.7, "capabilities": ['chat', 'agentic', 'code', 'coding', 'tools']},
    {"tag": "llama-4:scout", "params_b": 109, "vram_gb": 66.1, "capabilities": ['tools', 'vision', 'chat', 'agentic']},
    {"tag": "llama-4:maverick", "params_b": 400, "vram_gb": 243.7, "capabilities": ['tools', 'vision', 'chat', 'agentic']},
    {"tag": "qwen2.5:3b", "params_b": 3.09, "vram_gb": 2.1, "capabilities": ['tools', 'chat', 'agentic']},
    {"tag": "qwen2.5:7b", "params_b": 7.62, "vram_gb": 4.9, "capabilities": ['chat', 'agentic', 'code', 'coding', 'tools']},
    {"tag": "qwen2.5:14b", "params_b": 14.7, "vram_gb": 9.7, "capabilities": ['chat', 'agentic', 'code', 'coding', 'tools']},
    {"tag": "qwen2.5:32b", "params_b": 32.5, "vram_gb": 20.8, "capabilities": ['chat', 'agentic', 'code', 'coding', 'tools']},
    {"tag": "qwen2.5:72b", "params_b": 72.7, "vram_gb": 48.6, "capabilities": ['chat', 'agentic', 'code', 'coding', 'tools']},
    {"tag": "qwen2.5-vl:3b", "params_b": 3.09, "vram_gb": 2.1, "capabilities": ['vision', 'chat']},
    {"tag": "qwen2.5-vl:7b", "params_b": 7.62, "vram_gb": 4.9, "capabilities": ['tools', 'vision', 'chat', 'agentic']},
    {"tag": "qwen2.5-vl:72b", "params_b": 72.7, "vram_gb": 48.6, "capabilities": ['tools', 'vision', 'chat', 'agentic']},
    {"tag": "qwen3:8b", "params_b": 8.2, "vram_gb": 5.6, "capabilities": ['reasoning', 'math', 'agentic', 'chat', 'tools']},
    {"tag": "qwen3:32b", "params_b": 32.8, "vram_gb": 20.8, "capabilities": ['reasoning', 'math', 'agentic', 'chat', 'tools']},
    {"tag": "qwen3-coder-30b:a3b", "params_b": 30.5, "vram_gb": 19.0, "capabilities": ['chat', 'agentic', 'code', 'coding', 'tools']},
    {"tag": "qwen3-next-80b:a3b", "params_b": 80, "vram_gb": 48.5, "capabilities": ['tools', 'chat', 'agentic']},
    # DeepSeek-R1 local — strong reasoning, coding; scaled by parameter count
    {"tag": "deepseek-r1:7b",  "params_b": 7.62, "vram_gb": 4.9,  "capabilities": ['reasoning', 'math', 'chat', 'code', 'coding'], "intelligence_score": 20},
    {"tag": "deepseek-r1:14b", "params_b": 14.7, "vram_gb": 9.7,  "capabilities": ['reasoning', 'math', 'chat', 'code', 'coding'], "intelligence_score": 24},
    {"tag": "deepseek-r1:32b", "params_b": 32.5, "vram_gb": 20.8, "capabilities": ['reasoning', 'math', 'chat', 'code', 'coding'], "intelligence_score": 27},
    {"tag": "deepseek-r1:70b", "params_b": 70.6, "vram_gb": 43.7, "capabilities": ['reasoning', 'math', 'chat', 'code', 'coding'], "intelligence_score": 27},
    {"tag": "qwq:32b", "params_b": 32.5, "vram_gb": 20.9, "capabilities": ['reasoning', 'math', 'agentic', 'chat', 'tools'], "intelligence_score": 30},
    {"tag": "phi-4-mini:3.8b", "params_b": 3.84, "vram_gb": 3.0, "capabilities": ['chat']},
    {"tag": "phi-4:14b", "params_b": 14.7, "vram_gb": 9.8, "capabilities": ['coding', 'chat', 'code']},
    {"tag": "mistral-nemo:12b", "params_b": 12.2, "vram_gb": 8.4, "capabilities": ['tools', 'chat', 'agentic']},
    {"tag": "mistral-small-3.1:24b", "params_b": 23.6, "vram_gb": 14.9, "capabilities": ['chat', 'agentic', 'code', 'coding', 'vision', 'tools']},
    {"tag": "gemma-2:9b", "params_b": 9.24, "vram_gb": 7.0, "capabilities": ['chat']},
    {"tag": "gemma-2:27b", "params_b": 27.2, "vram_gb": 18.1, "capabilities": ['chat']},
    {"tag": "gemma-3:4b", "params_b": 3.88, "vram_gb": 3.1, "capabilities": ['vision', 'chat']},
    {"tag": "gemma-3:12b", "params_b": 11.8, "vram_gb": 9.2, "capabilities": ['coding', 'vision', 'chat', 'code']},
    {"tag": "gemma-3:27b", "params_b": 27.0, "vram_gb": 18.9, "capabilities": ['coding', 'vision', 'chat', 'code']},
    {"tag": "glm-4.7:flash", "params_b": 31, "vram_gb": 18.5, "capabilities": ['reasoning', 'math', 'agentic', 'chat', 'tools']},
    {"tag": "nemotron-nano:12b", "params_b": 12.6, "vram_gb": 7.5, "capabilities": ['reasoning', 'math', 'agentic', 'chat', 'tools']},
    # Qwen2.5-Coder — top local coding models (LiveCodeBench competitive)
    {"tag": "qwen2.5-coder:7b",  "params_b": 7.62, "vram_gb": 4.9,  "capabilities": ['coding', 'chat', 'code', 'agentic', 'tools'], "intelligence_score": 18},
    {"tag": "qwen2.5-coder:14b", "params_b": 14.7, "vram_gb": 9.7,  "capabilities": ['coding', 'chat', 'code', 'agentic', 'tools'], "intelligence_score": 22},
    {"tag": "qwen2.5-coder:32b", "params_b": 32.5, "vram_gb": 20.8, "capabilities": ['coding', 'chat', 'code', 'agentic', 'tools'], "intelligence_score": 26},
    {"tag": "codestral:22b", "params_b": 22.2, "vram_gb": 14.2, "capabilities": ['chat']},
    {"tag": "starcoder2:15b", "params_b": 15.6, "vram_gb": 10.2, "capabilities": ['coding', 'chat', 'code']},
    {"tag": "phi-3-mini:3.8b", "params_b": 3.82, "vram_gb": 3.9, "capabilities": ['chat']},
    {"tag": "deepseek-coder:6.7b", "params_b": 6.74, "vram_gb": 6.0, "capabilities": ['coding', 'chat', 'code']},
    {"tag": "smallthinker:3b", "params_b": 3.09, "vram_gb": 2.1, "capabilities": ['reasoning', 'math', 'chat']},
    {"tag": "ministral:8b", "params_b": 8.02, "vram_gb": 5.6, "capabilities": ['tools', 'chat', 'agentic']},
    {"tag": "devstral-small:24b", "params_b": 23.6, "vram_gb": 14.9, "capabilities": ['chat', 'agentic', 'code', 'coding', 'tools']},
    {"tag": "magistral-small:24b", "params_b": 23.6, "vram_gb": 14.9, "capabilities": ['reasoning', 'math', 'agentic', 'chat', 'vision', 'tools']},
    {"tag": "gpt-oss:20b", "params_b": 21, "vram_gb": 11.9, "capabilities": ['reasoning', 'math', 'agentic', 'chat', 'tools']},
    {"tag": "minimax:m2.1", "params_b": 229, "vram_gb": 138.3, "capabilities": ['reasoning', 'math', 'agentic', 'chat', 'code', 'coding', 'tools']},
    {"tag": "step-3.5:flash", "params_b": 197, "vram_gb": 119.7, "capabilities": ['reasoning', 'math', 'agentic', 'chat', 'code', 'coding', 'tools']},
    {"tag": "glm:4.6", "params_b": 357, "vram_gb": 216.3, "capabilities": ['reasoning', 'math', 'agentic', 'chat', 'tools']},
    # Qwen3.5 family — strong reasoning+coding, 262k context, MoE variants efficient
    {"tag": "qwen3.5:0.8b",  "params_b": 0.8,   "vram_gb": 0.6,  "ctx_k": 262, "capabilities": ['chat', 'reasoning', 'tools'], "intelligence_score": 11},
    {"tag": "qwen3.5:2b",    "params_b": 2.0,   "vram_gb": 1.3,  "ctx_k": 262, "capabilities": ['chat', 'reasoning', 'tools', 'coding'], "intelligence_score": 18},
    {"tag": "qwen3.5:4b",    "params_b": 4.0,   "vram_gb": 2.9,  "ctx_k": 262, "capabilities": ['chat', 'reasoning', 'tools', 'coding', 'agentic'], "intelligence_score": 27},
    {"tag": "qwen3.5:9b",    "params_b": 9.0,   "vram_gb": 5.9,  "ctx_k": 262, "capabilities": ['chat', 'reasoning', 'tools', 'coding', 'agentic', 'math'], "intelligence_score": 32},
    {"tag": "qwen3.5:27b",   "params_b": 27.0,  "vram_gb": 17.5, "ctx_k": 262, "capabilities": ['chat', 'reasoning', 'tools', 'coding', 'agentic', 'math'], "intelligence_score": 40},
    {"tag": "qwen3.5-35b:a3b",   "params_b": 35.0,  "vram_gb": 23.0, "ctx_k": 262, "capabilities": ['chat', 'reasoning', 'tools', 'coding', 'agentic', 'math', 'fast_response'], "intelligence_score": 44},
    {"tag": "qwen3.5-122b:a10b", "params_b": 122.0, "vram_gb": 26.6, "ctx_k": 262, "capabilities": ['chat', 'reasoning', 'tools', 'coding', 'agentic', 'math', 'fast_response', 'long_context'], "intelligence_score": 50},
    {"tag": "qwen3.5-397b:a17b", "params_b": 397.0, "vram_gb": 29.2, "ctx_k": 262, "capabilities": ['chat', 'reasoning', 'tools', 'coding', 'agentic', 'math', 'fast_response', 'long_context'], "intelligence_score": 54},
    {"tag": "minimax:m2.5", "params_b": 56.85, "vram_gb": 26.6, "capabilities": ['tools', 'chat', 'agentic']},
    {"tag": "glm:5", "params_b": 56.41, "vram_gb": 26.3, "capabilities": ['tools', 'chat', 'agentic']},
    {"tag": "gemma-4:e2b", "params_b": 2.3, "vram_gb": 3.4, "capabilities": ['reasoning', 'math', 'chat', 'agentic', 'vision', 'tools']},
    {"tag": "gemma-4:e4b", "params_b": 4.5, "vram_gb": 5.4, "capabilities": ['reasoning', 'math', 'chat', 'agentic', 'vision', 'tools']},
    {"tag": "gemma-4-26b:a4b", "params_b": 26.0, "vram_gb": 15.9, "capabilities": ['reasoning', 'math', 'chat', 'agentic', 'vision', 'tools']},
    {"tag": "gemma-4:31b", "params_b": 31.0, "vram_gb": 18.6, "capabilities": ['reasoning', 'math', 'chat', 'agentic', 'vision', 'tools']},
    {"tag": "nemotron-3-nano:4b", "params_b": 3.97, "vram_gb": 2.8, "capabilities": ['reasoning', 'math', 'agentic', 'chat', 'tools']},
    {"tag": "nemotron-cascade-2-30b:a3b", "params_b": 30.0, "vram_gb": 23.1, "capabilities": ['reasoning', 'math', 'agentic', 'chat', 'tools']},
]

def _dict_to_profile(d: dict) -> ModelProfile:
    """Convert a whatmodels-style dict entry to a ModelProfile."""
    tag = d.get("tag", d.get("name", "unknown"))
    name = tag
    params = float(d.get("params_b", 0.0))
    vram = float(d.get("vram_gb", params * 0.65))
    caps = d.get("capabilities", ["chat"])

    family = "unknown"
    tl = tag.lower()
    for fam in ("llama", "qwen", "gemma", "mistral", "phi", "deepseek", "nomic"):
        if fam in tl:
            family = fam
            break

    # ctx_k from dict (in thousands) or infer from tag pattern
    raw_ctx_k = d.get("ctx_k", None)
    if raw_ctx_k is None:
        tl2 = tag.lower()
        if "qwen3.5" in tl2 or "qwen3-5" in tl2:
            raw_ctx_k = 262
        elif "llama-4" in tl2 or "llama4" in tl2:
            raw_ctx_k = 1000
        elif "gemma-4" in tl2 or "gemma4" in tl2:
            raw_ctx_k = 128
        else:
            raw_ctx_k = 128

    return ModelProfile(
        name=name,
        family=family,
        params_b=params,
        quant="Q4_K_M",
        vram_gb=vram,
        ram_gb=vram * 2,
        ctx_k=int(raw_ctx_k),
        tok_per_sec_gpu={},
        tok_per_sec_cpu=1.5,
        capabilities=caps,
        recommended_for=caps,
        ollama_tag=tag,
        notes="",
        weight_gb=d.get("weight_gb", 0.0),
        kv_per_1k_gb=d.get("kv_per_1k_gb", 0.0),
        layers=d.get("layers") or 0,
        intelligence_score=float(d.get("intelligence_score", 0.0)),
    )

FALLBACK_MODELS: list[ModelProfile] = [_dict_to_profile(d) for d in MODELS]

async def inspect_local_models(host: str = "localhost:11434") -> list[dict]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"http://{host}/api/tags")
            data = resp.json()
            models_data = []
            for model in data.get("models", []):
                name = model.get("name", "unknown")
                size_bytes = model.get("size", 0)
                models_data.append({"name": name, "size_bytes": size_bytes})
            return models_data
        except Exception:
            return []

async def build_model_catalog(device_profile, host: str = "localhost:11434", show_progress: bool = False) -> list[ModelProfile]:
    local_models = await inspect_local_models(host)
    installed_names = {m["name"] for m in local_models}

    result = []
    for model in FALLBACK_MODELS:
        is_inst = model.name in installed_names
        vram_est = 0.0

        if is_inst:
            for local in local_models:
                if local["name"] == model.name:
                    size_bytes = local.get("size_bytes", 0)
                    if size_bytes > 0:
                        vram_est = estimate_vram_from_size(size_bytes)
                    break

        new_model = ModelProfile(
            name=model.name,
            family=model.family,
            params_b=model.params_b,
            quant=model.quant,
            vram_gb=model.vram_gb,
            ram_gb=model.ram_gb,
            ctx_k=model.ctx_k,
            tok_per_sec_gpu=model.tok_per_sec_gpu,
            tok_per_sec_cpu=model.tok_per_sec_cpu,
            capabilities=model.capabilities,
            recommended_for=model.recommended_for,
            ollama_tag=model.ollama_tag,
            notes=model.notes,
            is_installed=is_inst,
            vram_estimated_gb=vram_est
        )
        result.append(new_model)
        
    # 2026 Addition: Include any locally installed models that are not in the hardcoded list
    cataloged_names = {m.name for m in FALLBACK_MODELS}
    for local in local_models:
        if local["name"] not in cataloged_names:
            size_bytes = local.get("size_bytes", 0)
            vram_est = estimate_vram_from_size(size_bytes)
            # Estimate params based on typical 4-bit quantization (~0.7 GB per 1B)
            params_est = round(vram_est / 0.7, 1)
            
            result.append(ModelProfile(
                name=local["name"],
                family="local",
                params_b=params_est,
                quant="unknown",
                vram_gb=vram_est,
                ram_gb=vram_est * 2,
                ctx_k=128,
                tok_per_sec_gpu={},
                tok_per_sec_cpu=1.5,
                capabilities=["chat"],
                recommended_for=["chat"],
                ollama_tag=local["name"],
                notes="Custom local model",
                is_installed=True,
                vram_estimated_gb=vram_est
            ))

    return result

def estimate_vram_from_size(size_bytes: float) -> float:
    gb_from_size = size_bytes / (1.073 * 1e9)
    q4_overhead = 1.08
    return gb_from_size * q4_overhead

from neuralbrok.hardware import lookup_gpu

def get_tok_per_sec(profile: ModelProfile, device_key: str, bandwidth: Optional[float] = None) -> float:
    """
    Calculate estimated tokens per second. 
    Priority: 
    1. Explicit bandwidth (if provided)
    2. Lookup GPU by device_key and use its bandwidth
    3. Profile's tok_per_sec_gpu map
    4. Profile's tok_per_sec_cpu (final fallback)
    """
    # 1. Use explicit bandwidth (most accurate if detected by telemetry)
    bw = bandwidth
    
    # 2. Lookup GPU if no bandwidth provided
    if not bw:
        gpu = lookup_gpu(device_key)
        if gpu and gpu.bandwidth_gbps:
            bw = gpu.bandwidth_gbps
            
    if bw:
        # Precision weight (whatmodels style) or fallback to params * 0.7 for 4-bit
        weight = profile.weight_gb if profile.weight_gb > 0 else (profile.params_b * 0.7)
        overhead = 1.0 # System overhead/KV cache baseline
        return round(bw / (weight + overhead), 1)

    # 3. Fallback to hardcoded tables
    device_key_low = device_key.lower()
    if device_key_low in profile.tok_per_sec_gpu:
        return float(profile.tok_per_sec_gpu[device_key_low])
    
    for k, v in profile.tok_per_sec_gpu.items():
        if k in device_key_low:
            return float(v)
            
    # 4. CPU Fallback
    return float(profile.tok_per_sec_cpu)

def get_runnable_models(vram_gb: float, ram_gb: float, device_key: str, is_laptop: bool = False, models: Optional[list[ModelProfile]] = None) -> list[ModelProfile]:
    if models is None:
        models = FALLBACK_MODELS

    headroom = 1.0 if is_laptop else 0.5
    usable_ram = ram_gb - 2.0
    usable_vram = vram_gb - headroom

    runnable = []
    for model in models:
        # Precision routing: assume 4k context for safety check, or use model's max if it's a tight fit
        vram_needed = estimate_vram_requirement(model, context_k=4) 
        
        if vram_gb > 0 and vram_needed > usable_vram and not model.is_installed:
            continue

        is_moe = "a" in model.name and "-" in model.name
        if is_moe:
            full_weight_ram = model.params_b * 0.65
            if full_weight_ram > usable_ram:
                continue

        if ram_gb > 0 and model.ram_gb > usable_ram:
            continue

        runnable.append(model)

    def quality_score(m: ModelProfile):
        ctx_mult = 1.0
        if m.ctx_k >= 32 and m.ctx_k < 128:
            ctx_mult = 1.2
        elif m.ctx_k >= 128:
            ctx_mult = 1.5
        return m.params_b * ctx_mult

    runnable.sort(key=quality_score, reverse=True)
    return runnable
