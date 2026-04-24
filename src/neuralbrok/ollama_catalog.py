"""
ollama_catalog.py — Live Ollama model library discovery.

Fetches the latest models from Ollama library and resolves them
against your hardware profile for smart model matching.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List

import httpx

logger = logging.getLogger(__name__)

# ── Ollama Cloud models (run via `ollama run model:cloud`) ────────────────────
# These run in Ollama's cloud infra, not locally — no VRAM needed.
OLLAMA_CLOUD_MODELS = [
    {
        "tag": "gpt-oss:120b-cloud",
        "name": "GPT-OSS 120B",
        "params_b": 120.0,
        "description": "Open-source frontier performance — exceptional for complex reasoning & logic",
        "capabilities": ["chat", "reasoning", "math", "coding"],
        "context_k": 128,
        "tier": "flagship",
    },
    {
        "tag": "qwen3-coder:480b-cloud",
        "name": "Qwen3 Coder 480B",
        "params_b": 480.0,
        "description": "State-of-the-art coding and agentic performance — full-scale MoE",
        "capabilities": ["chat", "coding", "reasoning", "tools", "agentic"],
        "context_k": 128,
        "tier": "flagship",
    },
    {
        "tag": "kimi-k2.6:cloud",
        "name": "Kimi K2.6",
        "params_b": 1000.0,
        "description": "Moonshot AI flagship — 1T MoE, exceptional instruction following",
        "capabilities": ["chat", "coding", "reasoning", "tools", "agentic"],
        "context_k": 128,
        "tier": "flagship",
    },
    {
        "tag": "deepseek-r1:cloud",
        "name": "DeepSeek R1",
        "params_b": 671.0,
        "description": "DeepSeek R1 full via Ollama Cloud — frontier reasoning and STEM",
        "capabilities": ["chat", "reasoning", "math", "coding"],
        "context_k": 128,
        "tier": "flagship",
    },
    {
        "tag": "llama4-scout:cloud",
        "name": "Llama 4 Scout",
        "params_b": 109.0,
        "description": "Meta Llama 4 Scout — frontier vision + tool use, massive context",
        "capabilities": ["chat", "coding", "vision", "tools"],
        "context_k": 128,
        "tier": "standard",
    },
]

# ── Popular local model catalog (fallback if Ollama registry is unreachable) ──
KNOWN_LOCAL_MODELS = [
    {"tag": "qwen3:7b",      "params_b": 7.0,  "vram_gb": 4.8,  "capabilities": ["chat", "tools", "reasoning", "multilingual"]},
    {"tag": "qwen3:14b",     "params_b": 14.0, "vram_gb": 9.5,  "capabilities": ["chat", "coding", "math", "reasoning"]},
    {"tag": "qwen3:32b",     "params_b": 32.0, "vram_gb": 21.0, "capabilities": ["chat", "coding", "math", "reasoning"]},
    {"tag": "qwen2.5-coder:7b","params_b": 7.0,  "vram_gb": 4.5,  "capabilities": ["chat", "coding", "tools"]},
    {"tag": "qwen3-coder:30b", "params_b": 30.0, "vram_gb": 20.0, "capabilities": ["chat", "coding", "reasoning", "agentic"]},
    {"tag": "deepseek-r1:7b",  "params_b": 7.0,  "vram_gb": 5.0,  "capabilities": ["chat", "reasoning", "math", "coding"]},
    {"tag": "deepseek-r1:32b", "params_b": 32.0, "vram_gb": 20.0, "capabilities": ["chat", "reasoning", "math", "coding"]},
    {"tag": "gemma4:9b",       "params_b": 9.0,  "vram_gb": 6.5,  "capabilities": ["chat", "vision", "tools"]},
    {"tag": "gemma4:26b",      "params_b": 26.0, "vram_gb": 18.0, "capabilities": ["chat", "vision", "reasoning"]},
    {"tag": "llama4:8b",       "params_b": 8.0,  "vram_gb": 5.5,  "capabilities": ["chat", "tools", "long_context"]},
    {"tag": "phi-4:14b",       "params_b": 14.0, "vram_gb": 9.2,  "capabilities": ["chat", "reasoning", "math"]},
    {"tag": "phi-4-mini:3.8b", "params_b": 3.8,  "vram_gb": 2.8,  "capabilities": ["chat", "coding", "reasoning"]},
    {"tag": "nomic-embed-text:v1.5","params_b":0.1,"vram_gb":0.3,"capabilities": ["embedding"]},
]


@dataclass
class OllamaModelEntry:
    tag: str
    name: str
    params_b: float
    vram_gb: float
    description: str
    capabilities: List[str]
    context_k: int = 32
    is_cloud: bool = False
    cloud_tag: Optional[str] = None
    weight_gb: float = 0.0
    kv_per_1k_gb: float = 0.0


def _estimate_vram(params_b: float, quant: str = "Q4_K_M") -> float:
    """Estimate VRAM requirement from parameter count and quantization."""
    bits_map = {"q4_k_m": 4.5, "q8_0": 8.0, "fp16": 16.0, "f16": 16.0, "q5_k_m": 5.5, "q6_k": 6.5}
    bits = bits_map.get(quant.lower(), 4.5)
    weight_gb = (params_b * 1e9 * bits / 8) / (1024 ** 3)
    return weight_gb * 1.12  # 12% overhead for KV cache and system memory


def get_trending_ollama_models() -> List[dict]:
    """
    Scrape the Ollama library for trending models.
    Uses BeautifulSoup for robust discovery if the search API is limited.
    """
    try:
        from bs4 import BeautifulSoup
        url = "https://ollama.com/library"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        resp = httpx.get(url, headers=headers, timeout=5.0)
        if resp.status_code != 200:
            return []
        
        soup = BeautifulSoup(resp.text, "html.parser")
        models = []
        # Ollama library items are typically in li tags with x-data
        for item in soup.select("li[x-data]"):
            name_el = item.select_one("h2")
            desc_el = item.select_one("p")
            pulls_el = item.select_one("[title*='Pull']")
            
            if name_el:
                name = name_el.text.strip()
                desc = desc_el.text.strip() if desc_el else ""
                pulls = pulls_el.get("title", "") if pulls_el else ""
                
                models.append({
                    "tag": name,
                    "name": name,
                    "description": desc,
                    "pulls": pulls,
                    "capabilities": _infer_capabilities(name, desc)
                })
        return models
    except Exception as e:
        logger.debug(f"Trending scrape failed: {e}")
        return []


def fetch_latest_ollama_models(timeout: float = 4.0) -> List[OllamaModelEntry]:
    """
    Fetch the latest trending/featured models from Ollama library API.
    Returns a curated list combining live data with our known catalog.
    Falls back to KNOWN_LOCAL_MODELS if the registry is unreachable.
    """
    entries = []

    # Try Ollama registry API first
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(
                "https://ollama.com/api/search?q=&p=1&sort=popular&limit=50",
                headers={"Accept": "application/json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                for m in models:
                    tag = m.get("name", "")
                    if not tag:
                        continue
                    # Pull metadata
                    size_b = m.get("parameter_size", "")
                    params_b = _parse_params(size_b)
                    desc = m.get("description", "")[:120]
                    caps = _infer_capabilities(tag, desc)
                    vram = _estimate_vram(params_b)
                    entries.append(OllamaModelEntry(
                        tag=tag,
                        name=m.get("title", tag),
                        params_b=params_b,
                        vram_gb=vram,
                        description=desc,
                        capabilities=caps,
                        context_k=_infer_context(m),
                    ))
                logger.info(f"Fetched {len(entries)} models from Ollama registry")
    except Exception as e:
        logger.debug(f"Ollama registry fetch failed, using known catalog: {e}")

    # Always merge with our curated catalog (deduplicating by base tag)
    existing_tags = {e.tag.split(":")[0] for e in entries}
    for m in KNOWN_LOCAL_MODELS:
        base = m["tag"].split(":")[0]
        if base not in existing_tags:
            entries.append(OllamaModelEntry(
                tag=m["tag"],
                name=m["tag"],
                params_b=m["params_b"],
                vram_gb=m["vram_gb"],
                description="",
                capabilities=m["capabilities"],
            ))
            existing_tags.add(base)

    return entries


def _parse_params(size_str: str) -> float:
    """Parse parameter count string like '8B', '70B', '235B-A22B'."""
    if not size_str:
        return 7.0
    m = re.search(r"(\d+\.?\d*)B", size_str, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return 7.0


def _infer_capabilities(tag: str, desc: str) -> List[str]:
    """Infer capabilities from model name and description."""
    caps = ["chat"]
    combined = (tag + " " + desc).lower()
    if any(x in combined for x in ["code", "coder", "coding", "programm"]):
        caps.append("coding")
    if any(x in combined for x in ["reason", "think", "r1", "r2", "qwq", "deepseek"]):
        caps.append("reasoning")
    if any(x in combined for x in ["vision", "vl", "visual", "image", "multimodal"]):
        caps.append("vision")
    if any(x in combined for x in ["math", "mathematic", "stem"]):
        caps.append("math")
    if any(x in combined for x in ["tool", "function", "agent", "agentic"]):
        caps.append("tools")
    if any(x in combined for x in ["embed", "embedding", "nomic", "mxbai", "bge"]):
        caps = ["embedding"]
    if any(x in combined for x in ["multilingual", "multi-lingual", "translation"]):
        caps.append("multilingual")
    return caps


def _infer_context(model_data: dict) -> int:
    """Infer context window from model metadata."""
    ctx = model_data.get("context_length", 0)
    if ctx >= 128000:
        return 128
    elif ctx >= 32000:
        return 32
    elif ctx >= 16000:
        return 16
    return 32


def get_runnable_local_models(
    vram_gb: float,
    entries: List[OllamaModelEntry],
    headroom_gb: float = 0.5,
) -> List[OllamaModelEntry]:
    """Filter entries to models that fit in available VRAM."""
    usable = vram_gb - headroom_gb
    return [m for m in entries if not m.is_cloud and m.vram_gb <= usable]


def get_cloud_recommendations(
    vram_gb: float,
    workload: List[str],
) -> List[dict]:
    """
    Returns Ollama Cloud model recommendations based on VRAM and workload.
    Always returns cloud models; scoring is workload-weighted.
    """
    scored = []
    for cm in OLLAMA_CLOUD_MODELS:
        score = 0.0
        for cap in cm["capabilities"]:
            if cap in workload:
                score += 20.0
        if cm["tier"] == "flagship":
            score += 10.0
        # Boost coding-focused models when workload includes coding
        if "coding" in workload and "coding" in cm["capabilities"]:
            score += 15.0
        scored.append((score, cm))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [cm for _, cm in scored[:5]]


def assess_hardware(vram_gb: float, bandwidth_gbps: Optional[float] = None) -> dict:
    """
    Assess whether hardware is suitable for local LLM inference.
    Returns a structured assessment with tier, message and recommendations.
    """
    if vram_gb == 0:
        # CPU only
        return {
            "tier": "cpu_only",
            "suitable_for_local": False,
            "message": "No GPU detected. Local inference will be very slow on CPU only.",
            "max_comfortable_model_gb": 0.0,
            "suggest_cloud": True,
            "speed_note": "CPU inference: typically 1-5 tok/s for quantized 7B models",
        }
    elif vram_gb < 4:
        return {
            "tier": "very_low",
            "suitable_for_local": False,
            "message": f"With {vram_gb:.1f}GB VRAM, only tiny models (0.6B–1.7B) run locally. Quality will be limited.",
            "max_comfortable_model_gb": vram_gb - 0.5,
            "suggest_cloud": True,
            "speed_note": "Can run small models but responses may feel slow and low-quality.",
        }
    elif vram_gb < 8:
        # Estimate speed if bandwidth known
        if bandwidth_gbps:
            est_tps_7b = bandwidth_gbps / (5.0 + 1.0)  # ~5GB Q4 7B model
            speed_note = f"~{est_tps_7b:.0f} tok/s estimated for a 7B Q4 model on your GPU"
        else:
            speed_note = "Good speed for models up to ~6B, slower for 7B+ models"

        return {
            "tier": "low",
            "suitable_for_local": True,
            "message": f"With {vram_gb:.1f}GB VRAM, you can run solid 4B–7B models. Quality is good for most tasks.",
            "max_comfortable_model_gb": vram_gb - 0.8,
            "suggest_cloud": True,  # suggest as an option, not required
            "speed_note": speed_note,
        }
    elif vram_gb < 12:
        if bandwidth_gbps:
            est_tps_7b = bandwidth_gbps / (5.0 + 1.0)
            speed_note = f"~{est_tps_7b:.0f} tok/s for a 7B Q4 model — snappy"
        else:
            speed_note = "Great speed for 7B–8B models, capable for 14B with quantization"

        return {
            "tier": "mid",
            "suitable_for_local": True,
            "message": f"With {vram_gb:.1f}GB VRAM, you can run 7B–8B models comfortably and some 14B quantized models.",
            "max_comfortable_model_gb": vram_gb - 0.8,
            "suggest_cloud": False,
            "speed_note": speed_note,
        }
    elif vram_gb < 24:
        return {
            "tier": "good",
            "suitable_for_local": True,
            "message": f"With {vram_gb:.1f}GB VRAM, you can run 13B–20B models smoothly. Very capable local setup.",
            "max_comfortable_model_gb": vram_gb - 1.0,
            "suggest_cloud": False,
            "speed_note": "Excellent local inference speed. Cloud mostly for spillover.",
        }
    else:
        return {
            "tier": "excellent",
            "suitable_for_local": True,
            "message": f"With {vram_gb:.1f}GB VRAM, you can run 30B–70B models. Near-frontier local quality.",
            "max_comfortable_model_gb": vram_gb - 1.5,
            "suggest_cloud": False,
            "speed_note": "Exceptional local speed. You're set.",
        }
