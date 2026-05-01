"""
Runtime provider management for NeuralBroker v2.0.

Enables adding, removing, testing providers at runtime without restart,
and auto-detecting local runtimes (Ollama, LM Studio, Docker Model Runner, etc).
"""
import logging
import os
from pathlib import Path
from typing import Optional

import httpx
import yaml

logger = logging.getLogger(__name__)

# Known local runtimes and their default endpoints
LOCAL_RUNTIMES = {
    "ollama":              {"url": "http://localhost:11434", "health": "/api/tags", "type": "local"},
    "llama_cpp":           {"url": "http://localhost:8080",  "health": "/health",   "type": "local"},
    "lm_studio":           {"url": "http://localhost:1234",  "health": "/v1/models","type": "local"},
    "docker_model_runner": {"url": "http://localhost:12434", "health": "/engines/llama.cpp/v1/models", "type": "local"},
    "localai":             {"url": "http://localhost:8080",  "health": "/readyz",   "type": "local"},
    "jan":                 {"url": "http://localhost:1337",  "health": "/v1/models","type": "local"},
}

# Known cloud providers
CLOUD_PROVIDERS = {
    "groq":       {"base_url": "https://api.groq.com/openai/v1",     "key_env": "GROQ_KEY",       "cost": 0.00006},
    "together":   {"base_url": "https://api.together.xyz/v1",        "key_env": "TOGETHER_KEY",    "cost": 0.00020},
    "openai":     {"base_url": "https://api.openai.com/v1",          "key_env": "OPENAI_API_KEY",  "cost": 0.00060},
    "anthropic":  {"base_url": "https://api.anthropic.com",          "key_env": "ANTHROPIC_API_KEY","cost": 0.00300},
    "deepseek":   {"base_url": "https://api.deepseek.com/v1",       "key_env": "DEEPSEEK_KEY",    "cost": 0.00014},
    "fireworks":  {"base_url": "https://api.fireworks.ai/inference/v1","key_env": "FIREWORKS_KEY", "cost": 0.00020},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",      "key_env": "OPENROUTER_KEY",  "cost": 0.00010},
    "cerebras":   {"base_url": "https://api.cerebras.ai/v1",        "key_env": "CEREBRAS_KEY",    "cost": 0.00010},
    "mistral":    {"base_url": "https://api.mistral.ai/v1",         "key_env": "MISTRAL_KEY",     "cost": 0.00025},
    "gemini":     {"base_url": "https://generativelanguage.googleapis.com/v1beta", "key_env": "GOOGLE_API_KEY", "cost": 0.00035},
}


def auto_detect_providers() -> dict[str, dict]:
    """Auto-detect available local runtimes and cloud providers.

    Returns dict of {name: {type, url, available, ...}}.
    """
    detected = {}

    # Check local runtimes
    for name, info in LOCAL_RUNTIMES.items():
        url = info["url"] + info["health"]
        available = _check_endpoint(url)
        detected[name] = {
            "type": "local",
            "url": info["url"],
            "available": available,
            "source": "auto-detect",
        }

    # Check cloud providers via env vars
    for name, info in CLOUD_PROVIDERS.items():
        key = os.environ.get(info["key_env"], "")
        detected[name] = {
            "type": "cloud",
            "url": info["base_url"],
            "available": bool(key),
            "api_key_env": info["key_env"],
            "has_key": bool(key),
            "cost_per_1k": info["cost"],
            "source": "env" if key else "not_configured",
        }

    return detected


def _check_endpoint(url: str, timeout: float = 1.0) -> bool:
    """Check if an endpoint is reachable."""
    try:
        import urllib.request
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


async def test_provider(name: str, base_url: str, api_key: str = "") -> dict:
    """Send a test request to verify a provider works."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    test_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Say 'ok'"}],
        "max_tokens": 5,
        "stream": False,
    }

    try:
        import time
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=test_body,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            if resp.status_code == 200:
                return {"success": True, "latency_ms": round(latency_ms, 1), "status": resp.status_code}
            return {"success": False, "latency_ms": round(latency_ms, 1), "status": resp.status_code, "error": resp.text[:200]}
    except Exception as e:
        return {"success": False, "error": str(e)}


def save_provider_to_config(
    name: str,
    provider_type: str,
    api_key_env: str = "",
    base_url: str = "",
    cost_per_1k: float = 0.0,
    config_path: Optional[Path] = None,
) -> Path:
    """Add or update a provider in config.yaml."""
    if config_path is None:
        config_path = Path.home() / ".neuralbrok" / "config.yaml"

    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config
    if config_path.exists():
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    if provider_type == "local":
        nodes = data.setdefault("local_nodes", [])
        # Update or add
        existing = [n for n in nodes if n.get("name") == name]
        if existing:
            existing[0].update({"host": base_url.replace("http://", ""), "runtime": name})
        else:
            host = base_url.replace("http://", "") if base_url else "localhost:11434"
            nodes.append({"name": name, "runtime": name, "host": host, "vram_threshold": 0.80})
    else:
        providers = data.setdefault("cloud_providers", [])
        existing = [p for p in providers if p.get("name") == name]
        if existing:
            existing[0].update({"api_key_env": api_key_env, "base_url": base_url, "cost_per_1k_tokens": cost_per_1k})
        else:
            entry = {"name": name, "api_key_env": api_key_env}
            if base_url: entry["base_url"] = base_url
            if cost_per_1k > 0: entry["cost_per_1k_tokens"] = cost_per_1k
            providers.append(entry)

    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return config_path


def remove_provider_from_config(name: str, config_path: Optional[Path] = None) -> bool:
    """Remove a provider from config.yaml."""
    if config_path is None:
        config_path = Path.home() / ".neuralbrok" / "config.yaml"

    if not config_path.exists():
        return False

    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    removed = False

    nodes = data.get("local_nodes", [])
    filtered = [n for n in nodes if n.get("name") != name]
    if len(filtered) < len(nodes):
        data["local_nodes"] = filtered
        removed = True

    providers = data.get("cloud_providers", [])
    filtered = [p for p in providers if p.get("name") != name]
    if len(filtered) < len(providers):
        data["cloud_providers"] = filtered
        removed = True

    if removed:
        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return removed
