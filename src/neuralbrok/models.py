from dataclasses import dataclass, field
import asyncio
import json
import time
from pathlib import Path
from typing import Optional
import httpx

@dataclass
class ModelProfile:
    name: str
    family: str
    params_b: float
    quant: str
    vram_gb: float
    ram_gb: float
    ctx_k: int
    tok_per_sec_gpu: dict
    tok_per_sec_cpu: float
    capabilities: list[str]
    recommended_for: list[str]
    ollama_tag: str
    notes: str
    is_installed: bool = False
    vram_estimated_gb: float = 0.0

MODELS = [
    # Qwen3 family
    ModelProfile("qwen3:0.6b",   "qwen3",    0.6,  "q4_K_M", 0.8,  2,   32,  {"rtx3060": 150, "rtx3070": 180, "rtx3080": 220, "rtx3090": 250, "rtx4060": 180, "rtx4070": 220, "rtx4080": 280, "rtx4090": 350, "m1": 100, "m1pro": 120, "m1max": 140, "m2": 120, "m2pro": 140, "m2max": 160, "m3": 140, "m3pro": 160, "m3max": 180, "m4": 160, "m4pro": 180, "m4max": 200, "rx6800xt": 180, "rx7900xtx": 250, "cpu": 12.0}, 12.0, ["chat","tools","code","reasoning"], ["chat","coding"], "qwen3:0.6b", "Smallest usable model — fits any hardware"),
    ModelProfile("qwen3:1.7b",   "qwen3",    1.7,  "q4_K_M", 1.5,  4,   32,  {"rtx3060": 90, "rtx3070": 110, "rtx3080": 140, "rtx3090": 160, "rtx4060": 110, "rtx4070": 140, "rtx4080": 180, "rtx4090": 220, "m1": 60, "m2": 75, "m3": 90, "rx6800xt": 110, "cpu": 8.0}, 8.0,  ["chat","tools","code","reasoning"], ["chat","coding"], "qwen3:1.7b", "Good quality for <2GB VRAM"),
    ModelProfile("qwen3:4b",     "qwen3",    4.0,  "q4_K_M", 3.0,  6,   32,  {"rtx3060": 55, "rtx3070": 65, "rtx3080": 85, "rtx3090": 100, "rtx4060": 65, "rtx4070": 85, "rtx4080": 110, "rtx4090": 140, "m1": 35, "m2": 45, "m3": 55, "rx6800xt": 65, "cpu": 5.0}, 5.0,  ["chat","tools","code","reasoning"], ["chat","coding","agentic"], "qwen3:4b", "Best quality under 4GB VRAM"),
    ModelProfile("qwen3:8b",     "qwen3",    8.0,  "q4_K_M", 5.5,  8,   128, {"rtx3060": 28, "rtx3070": 35, "rtx3080": 42, "rtx3090": 50, "rtx4060": 35, "rtx4070": 45, "rtx4080": 60, "rtx4090": 80, "m1": 20, "m1pro": 25, "m1max": 35, "m2": 25, "m2pro": 30, "m2max": 40, "m3": 30, "m3pro": 35, "m3max": 45, "m4": 35, "m4pro": 40, "m4max": 50, "rx6800xt": 35, "rx7900xtx": 55, "cpu": 3.0}, 3.0,  ["chat","tools","code","reasoning","multilingual"], ["chat","coding","agentic","rag"], "qwen3:8b", "Best all-rounder for 6-8GB VRAM"),
    ModelProfile("qwen3:14b",    "qwen3",    14.0, "q4_K_M", 9.0,  16,  128, {"rtx3060": 15, "rtx3070": 20, "rtx3080": 25, "rtx3090": 32, "rtx4060": 20, "rtx4070": 26, "rtx4080": 35, "rtx4090": 48, "m1max": 20, "m2max": 24, "m3max": 28, "m4max": 32, "rx6800xt": 20, "rx7900xtx": 35, "cpu": 1.5}, 1.5,  ["chat","tools","code","reasoning","multilingual"], ["chat","coding","agentic","rag","math"], "qwen3:14b", "Strong reasoning for 10-12GB VRAM"),
    ModelProfile("qwen3:32b",    "qwen3",    32.0, "q4_K_M", 20.0, 32,  128, {"rtx3090": 15, "rtx4080": 16, "rtx4090": 22, "m2max": 10, "m3max": 12, "m4max": 14, "rx7900xtx": 16, "cpu": 0.5}, 0.5,  ["chat","tools","code","reasoning","multilingual"], ["chat","coding","agentic","math"], "qwen3:32b", "Near-frontier quality for 24GB VRAM"),
    ModelProfile("qwen3:30b-a3b","qwen3",    30.0, "q4_K_M", 4.0,  8,   128, {"rtx3060": 35, "rtx3070": 42, "rtx3080": 55, "rtx3090": 65, "rtx4060": 42, "rtx4070": 55, "rtx4080": 75, "rtx4090": 95, "m1max": 30, "m2max": 35, "m3max": 40, "m4max": 45, "rx6800xt": 42, "rx7900xtx": 65, "cpu": 2.0}, 2.0,  ["chat","tools","code","reasoning"], ["chat","coding","agentic"], "qwen3:30b-a3b", "MoE — frontier quality at 4GB active VRAM"),
    ModelProfile("qwen3:235b-a22b","qwen3",  235.0,"q4_K_M", 28.0, 64,  128, {"rtx3090": 8, "rtx4090": 12, "m2max": 4, "m3max": 5, "m4max": 6, "rx7900xtx": 9, "cpu": 0.1}, 0.1,  ["chat","tools","code","reasoning","multilingual"], ["chat","coding","math"], "qwen3:235b-a22b", "MoE — best open model, needs 24GB+ VRAM"),

    # Llama 4 family
    ModelProfile("llama4:scout", "llama4",   17.0, "q4_K_M", 8.0,  16,  128, {"rtx3060": 18, "rtx3070": 22, "rtx3080": 28, "rtx3090": 35, "rtx4060": 22, "rtx4070": 28, "rtx4080": 38, "rtx4090": 52, "m1max": 22, "m2max": 26, "m3max": 30, "m4max": 34, "rx6800xt": 22, "rx7900xtx": 38, "cpu": 2.0}, 2.0,  ["chat","tools","vision","code"], ["chat","coding","vision","agentic"], "llama4:scout", "Best vision+tools combo for 8-10GB VRAM"),
    ModelProfile("llama4:maverick","llama4", 17.0, "q4_K_M", 12.0, 24,  128, {"rtx3060": 16, "rtx3070": 20, "rtx3080": 25, "rtx3090": 32, "rtx4060": 20, "rtx4070": 25, "rtx4080": 35, "rtx4090": 48, "m1max": 20, "m2max": 24, "m3max": 28, "m4max": 32, "rx6800xt": 20, "rx7900xtx": 35, "cpu": 1.5}, 1.5,  ["chat","tools","vision","code"], ["chat","coding","vision","agentic"], "llama4:maverick", "Higher quality Llama 4 for 12-16GB VRAM"),

    # DeepSeek family
    ModelProfile("deepseek-r1:7b",  "deepseek", 7.0,  "q4_K_M", 5.0,  8,  64, {"rtx3060": 30, "rtx3070": 38, "rtx3080": 45, "rtx3090": 55, "rtx4060": 38, "rtx4070": 48, "rtx4080": 65, "rtx4090": 85, "m1max": 30, "m2max": 35, "m3max": 40, "m4max": 45, "rx6800xt": 38, "rx7900xtx": 60, "cpu": 3.0}, 3.0, ["chat","reasoning","code","math"], ["math","coding","reasoning"], "deepseek-r1:7b", "Best reasoning for 6GB VRAM"),
    ModelProfile("deepseek-r1:14b", "deepseek", 14.0, "q4_K_M", 9.0,  16, 64, {"rtx3060": 16, "rtx3070": 20, "rtx3080": 26, "rtx3090": 33, "rtx4060": 20, "rtx4070": 27, "rtx4080": 36, "rtx4090": 50, "m1max": 20, "m2max": 25, "m3max": 29, "m4max": 34, "rx6800xt": 20, "rx7900xtx": 36, "cpu": 1.5}, 1.5, ["chat","reasoning","code","math"], ["math","coding","reasoning"], "deepseek-r1:14b", "Strong math and code for 10-12GB VRAM"),
    ModelProfile("deepseek-r1:32b", "deepseek", 32.0, "q4_K_M", 20.0, 32, 64, {"rtx3090": 15, "rtx4080": 16, "rtx4090": 22, "m2max": 10, "m3max": 12, "m4max": 14, "rx7900xtx": 16, "cpu": 0.5}, 0.5, ["chat","reasoning","code","math"], ["math","coding","reasoning"], "deepseek-r1:32b", "Near-frontier reasoning for 24GB VRAM"),
    ModelProfile("deepseek-coder-v2:16b","deepseek",16.0,"q4_K_M",10.0,16,128,{"rtx3060": 14, "rtx3070": 18, "rtx3080": 23, "rtx3090": 30, "rtx4060": 18, "rtx4070": 24, "rtx4080": 32, "rtx4090": 45, "m1max": 18, "m2max": 22, "m3max": 26, "m4max": 30, "rx6800xt": 18, "rx7900xtx": 32, "cpu": 1.2}, 1.2, ["chat","code","tools"],            ["coding","agentic"],           "deepseek-coder-v2:16b", "Best dedicated coder for 10-12GB VRAM"),

    # Mistral family
    ModelProfile("mistral:7b",      "mistral",  7.0,  "q4_K_M", 5.0,  8,  32, {"rtx3060": 32, "rtx3070": 40, "rtx3080": 48, "rtx3090": 58, "rtx4060": 40, "rtx4070": 50, "rtx4080": 68, "rtx4090": 90, "m1max": 32, "m2max": 38, "m3max": 44, "m4max": 50, "rx6800xt": 40, "rx7900xtx": 65, "cpu": 3.0}, 3.0, ["chat","tools","code"],            ["chat","rag"],                 "mistral:7b", "Fast and reliable general chat"),
    ModelProfile("mistral-small:22b","mistral", 22.0, "q4_K_M", 14.0, 24, 32, {"rtx3080": 16, "rtx3090": 22, "rtx4070": 18, "rtx4080": 24, "rtx4090": 32, "m2max": 14, "m3max": 16, "m4max": 18, "rx7900xtx": 24, "cpu": 1.0}, 1.0, ["chat","tools","code"],            ["chat","rag","coding"],        "mistral-small:22b", "Strong instruction following for 16GB VRAM"),

    # Phi-4 family
    ModelProfile("phi4:14b",     "phi4",     14.0, "q4_K_M", 9.0,  16, 16, {"rtx3060": 16, "rtx3070": 20, "rtx3080": 26, "rtx3090": 33, "rtx4060": 20, "rtx4070": 27, "rtx4080": 36, "rtx4090": 50, "m1max": 20, "m2max": 25, "m3max": 29, "m4max": 34, "rx6800xt": 20, "rx7900xtx": 36, "cpu": 1.5}, 1.5,  ["chat","code","reasoning","math"],  ["coding","math","reasoning"],  "phi4:14b", "Microsoft — exceptional at math and code for size"),
    ModelProfile("phi4-mini:3.8b","phi4",    3.8,  "q4_K_M", 3.0,  6,  16, {"rtx3060": 60, "rtx3070": 72, "rtx3080": 90, "rtx3090": 110, "rtx4060": 72, "rtx4070": 92, "rtx4080": 120, "rtx4090": 150, "m1max": 45, "m2max": 55, "m3max": 65, "m4max": 75, "rx6800xt": 72, "rx7900xtx": 110, "cpu": 4.0}, 4.0,  ["chat","code","reasoning"],         ["coding","chat"],              "phi4-mini:3.8b", "Strong coder under 4GB VRAM"),

    # Gemma 3 family
    ModelProfile("gemma3:4b",    "gemma3",   4.0,  "q4_K_M", 3.0,  6,  128, {"rtx3060": 58, "rtx3070": 68, "rtx3080": 88, "rtx3090": 105, "rtx4060": 68, "rtx4070": 88, "rtx4080": 115, "rtx4090": 145, "m1max": 40, "m2max": 50, "m3max": 60, "m4max": 70, "rx6800xt": 68, "rx7900xtx": 105, "cpu": 5.0}, 5.0, ["chat","vision","tools"],           ["chat","vision"],              "gemma3:4b", "Google — vision capable under 4GB VRAM"),
    ModelProfile("gemma3:12b",   "gemma3",   12.0, "q4_K_M", 8.0,  16, 128, {"rtx3060": 18, "rtx3070": 24, "rtx3080": 30, "rtx3090": 38, "rtx4060": 24, "rtx4070": 30, "rtx4080": 42, "rtx4090": 55, "m1max": 24, "m2max": 28, "m3max": 34, "m4max": 40, "rx6800xt": 24, "rx7900xtx": 42, "cpu": 2.0}, 2.0, ["chat","vision","tools"],           ["chat","vision","rag"],        "gemma3:12b", "Good vision and chat for 8-10GB VRAM"),
    ModelProfile("gemma3:27b",   "gemma3",   27.0, "q4_K_M", 18.0, 32, 128, {"rtx3090": 18, "rtx4080": 18, "rtx4090": 25, "m2max": 12, "m3max": 14, "m4max": 16, "rx7900xtx": 18, "cpu": 0.8}, 0.8, ["chat","vision","tools"],           ["chat","vision"],              "gemma3:27b", "Best Gemma quality for 20GB+ VRAM"),

    # Nomic embedding
    ModelProfile("nomic-embed-text:v1.5","nomic",0.137,"f16",0.3,2,8, {"rtx3060": 800, "rtx3070": 900, "rtx3080": 1100, "rtx3090": 1300, "rtx4060": 900, "rtx4070": 1100, "rtx4080": 1400, "rtx4090": 1800, "m1max": 600, "m2max": 700, "m3max": 800, "m4max": 900, "rx6800xt": 900, "rx7900xtx": 1400, "cpu": 50.0}, 50.0, ["embedding"],                              ["rag","embedding"],            "nomic-embed-text:v1.5", "Best local embedding model for RAG"),
]

FALLBACK_MODELS = MODELS

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

    return result

def estimate_vram_from_size(size_bytes: float) -> float:
    gb_from_size = size_bytes / (1.073 * 1e9)
    q4_overhead = 1.08
    return gb_from_size * q4_overhead

def get_tok_per_sec(profile: ModelProfile, device_key: str) -> float:
    device_key = device_key.lower()
    if device_key in profile.tok_per_sec_gpu:
        return float(profile.tok_per_sec_gpu[device_key])
    
    for k, v in profile.tok_per_sec_gpu.items():
        if k in device_key:
            return float(v)
            
    return float(profile.tok_per_sec_cpu)

def get_runnable_models(vram_gb: float, ram_gb: float, device_key: str, is_laptop: bool = False, models: Optional[list[ModelProfile]] = None) -> list[ModelProfile]:
    if models is None:
        models = FALLBACK_MODELS

    headroom = 1.0 if is_laptop else 0.5
    usable_ram = ram_gb - 2.0
    usable_vram = vram_gb - headroom

    runnable = []
    for model in models:
        effective_vram = model.vram_estimated_gb if model.vram_estimated_gb > 0 else model.vram_gb

        if vram_gb > 0 and effective_vram > usable_vram:
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
