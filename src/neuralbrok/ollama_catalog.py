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
    {"tag": "deepseek-r1:7b", "params_b": 7.62, "vram_gb": 4.9, "capabilities": ['reasoning', 'math', 'chat', 'code', 'coding']},
    {"tag": "deepseek-r1:14b", "params_b": 14.7, "vram_gb": 9.7, "capabilities": ['reasoning', 'math', 'chat', 'code', 'coding']},
    {"tag": "deepseek-r1:32b", "params_b": 32.5, "vram_gb": 20.8, "capabilities": ['reasoning', 'math', 'chat', 'code', 'coding']},
    {"tag": "deepseek-r1:70b", "params_b": 70.6, "vram_gb": 43.7, "capabilities": ['reasoning', 'math', 'chat', 'code', 'coding']},
    {"tag": "qwq:32b", "params_b": 32.5, "vram_gb": 20.9, "capabilities": ['reasoning', 'math', 'agentic', 'chat', 'tools']},
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
    {"tag": "qwen2.5-coder:7b", "params_b": 7.62, "vram_gb": 4.9, "capabilities": ['coding', 'chat', 'code']},
    {"tag": "qwen2.5-coder:14b", "params_b": 14.7, "vram_gb": 9.7, "capabilities": ['coding', 'chat', 'code']},
    {"tag": "qwen2.5-coder:32b", "params_b": 32.5, "vram_gb": 20.8, "capabilities": ['coding', 'chat', 'code']},
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
    {"tag": "qwen3.5:0.8b", "params_b": 0.8, "vram_gb": 0.6, "capabilities": ['chat']},
    {"tag": "qwen3.5:2b", "params_b": 2.0, "vram_gb": 1.3, "capabilities": ['chat']},
    {"tag": "qwen3.5:4b", "params_b": 4.0, "vram_gb": 2.9, "capabilities": ['chat']},
    {"tag": "qwen3.5:9b", "params_b": 9.0, "vram_gb": 5.9, "capabilities": ['chat']},
    {"tag": "qwen3.5:27b", "params_b": 27.0, "vram_gb": 17.5, "capabilities": ['chat']},
    {"tag": "qwen3.5-35b:a3b", "params_b": 35.0, "vram_gb": 23.0, "capabilities": ['chat']},
    {"tag": "qwen3.5-122b:a10b", "params_b": 122.0, "vram_gb": 26.6, "capabilities": ['chat']},
    {"tag": "qwen3.5-397b:a17b", "params_b": 397.0, "vram_gb": 29.2, "capabilities": ['chat']},
    {"tag": "minimax:m2.5", "params_b": 56.85, "vram_gb": 26.6, "capabilities": ['tools', 'chat', 'agentic']},
    {"tag": "glm:5", "params_b": 56.41, "vram_gb": 26.3, "capabilities": ['tools', 'chat', 'agentic']},
    {"tag": "gemma-4:e2b", "params_b": 2.3, "vram_gb": 3.4, "capabilities": ['reasoning', 'math', 'chat', 'agentic', 'vision', 'tools']},
    {"tag": "gemma-4:e4b", "params_b": 4.5, "vram_gb": 5.4, "capabilities": ['reasoning', 'math', 'chat', 'agentic', 'vision', 'tools']},
    {"tag": "gemma-4-26b:a4b", "params_b": 26.0, "vram_gb": 15.9, "capabilities": ['reasoning', 'math', 'chat', 'agentic', 'vision', 'tools']},
    {"tag": "gemma-4:31b", "params_b": 31.0, "vram_gb": 18.6, "capabilities": ['reasoning', 'math', 'chat', 'agentic', 'vision', 'tools']},
    {"tag": "nemotron-3-nano:4b", "params_b": 3.97, "vram_gb": 2.8, "capabilities": ['reasoning', 'math', 'agentic', 'chat', 'tools']},
    {"tag": "nemotron-cascade-2-30b:a3b", "params_b": 30.0, "vram_gb": 23.1, "capabilities": ['reasoning', 'math', 'agentic', 'chat', 'tools']},
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
    is_low_bandwidth = bandwidth_gbps is not None and bandwidth_gbps < 150.0

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
            speed_note = f"~{est_tps_7b:.0f} tok/s estimated for a 7B Q4 model on your hardware"
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
            speed_note = f"~{est_tps_7b:.0f} tok/s for a 7B Q4 model" + (" — snappy" if est_tps_7b > 15 else " — playable but slow")
        else:
            speed_note = "Great speed for 7B–8B models, capable for 14B with quantization"

        return {
            "tier": "mid",
            "suitable_for_local": True,
            "message": f"With {vram_gb:.1f}GB VRAM, you can run 7B–8B models comfortably.",
            "max_comfortable_model_gb": (vram_gb - 0.8) if not is_low_bandwidth else 8.0,
            "suggest_cloud": is_low_bandwidth,
            "speed_note": speed_note,
        }
    elif vram_gb < 24:
        speed_note = "Excellent local inference speed. Cloud mostly for spillover."
        if bandwidth_gbps:
            est_tps_14b = bandwidth_gbps / (10.0 + 1.0)
            speed_note = f"~{est_tps_14b:.0f} tok/s estimated for a 14B model"

        return {
            "tier": "good",
            "suitable_for_local": True,
            "message": f"With {vram_gb:.1f}GB VRAM, you can run 13B–20B models. Very capable local setup.",
            "max_comfortable_model_gb": (vram_gb - 1.0) if not is_low_bandwidth else 10.0,
            "suggest_cloud": False,
            "speed_note": speed_note,
        }
    else:
        speed_note = "Exceptional local speed. You're set."
        if bandwidth_gbps:
            est_tps_32b = bandwidth_gbps / (22.0 + 1.0)
            speed_note = f"~{est_tps_32b:.0f} tok/s estimated for a 32B model"

        return {
            "tier": "excellent",
            "suitable_for_local": True,
            "message": f"With {vram_gb:.1f}GB VRAM, you can run 30B–70B models. Near-frontier local quality.",
            "max_comfortable_model_gb": (vram_gb - 1.5) if not is_low_bandwidth else 12.0,
            "suggest_cloud": False,
            "speed_note": speed_note,
        }

