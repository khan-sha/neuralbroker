"""
FastAPI application — NeuralBroker.

VRAM-aware LLM routing proxy with OpenAI-compatible endpoints,
internal observability APIs, and Prometheus metrics.
"""
import json
import logging
import os
import re
import time
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, Response, HTMLResponse, FileResponse

from neuralbrok.config import load_config, Config
from neuralbrok.types import OpenAIRequest, PolicyMode
from neuralbrok.router import PolicyEngine, route_request
from neuralbrok.telemetry import VramPoller
from neuralbrok.models import resolve_model

from neuralbrok import metrics as nb_metrics
from neuralbrok.providers import (
    BaseProvider,
    ProviderError,
    OllamaProvider,
    GroqProvider,
    TogetherProvider,
    OpenAIProvider,
    LlamaCppProvider,
    CerebrasProvider,
    DeepInfraProvider,
    FireworksProvider,
    LeptonProvider,
    NovitaProvider,
    HyperbolicProvider,
    MistralProvider,
    KimiProvider,
    DeepSeekProvider,
    QwenProvider,
    YiProvider,
    BaichuanProvider,
    ZhipuProvider,
    PerplexityProvider,
    AI21Provider,
    OctoAIProvider,
    OpenRouterProvider,
    AnthropicProvider,
    GeminiProvider,
    CohereProvider,
    ReplicateProvider,
    CloudflareProvider,
    BedrockProvider,
    AzureOpenAIProvider,
    VertexProvider,
)

# Dynamic provider registry — maps config name → provider class
PROVIDER_REGISTRY: dict[str, type] = {
    "groq": GroqProvider,
    "together": TogetherProvider,
    "openai": OpenAIProvider,
    "cerebras": CerebrasProvider,
    "deepinfra": DeepInfraProvider,
    "fireworks": FireworksProvider,
    "lepton": LeptonProvider,
    "novita": NovitaProvider,
    "hyperbolic": HyperbolicProvider,
    "mistral": MistralProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "cohere": CohereProvider,
    "kimi": KimiProvider,
    "deepseek": DeepSeekProvider,
    "qwen": QwenProvider,
    "yi": YiProvider,
    "baichuan": BaichuanProvider,
    "zhipu": ZhipuProvider,
    "perplexity": PerplexityProvider,
    "ai21": AI21Provider,
    "replicate": ReplicateProvider,
    "octoai": OctoAIProvider,
    "openrouter": OpenRouterProvider,
    "cloudflare": CloudflareProvider,
    "bedrock": BedrockProvider,
    "azure_openai": AzureOpenAIProvider,
    "vertex": VertexProvider,
}

# Legacy imports for backward compatibility
from neuralbrok.adapter import OllamaBackend, GroqBackend

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── Global State ──────────────────────────────────────────────────────────────

config: Optional[Config] = None
providers: dict[str, BaseProvider] = {}
provider_types: dict[str, str] = {}
provider_costs: dict[str, float] = {}
policy_engine: Optional[PolicyEngine] = None
poller: Optional[VramPoller] = None

# Legacy backend dict for backward compat with old adapter.py
backends: dict = {}





# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context for startup/shutdown."""
    global config, providers, provider_types, provider_costs
    global policy_engine, poller, backends

    user_config_path = os.path.expanduser("~/.neuralbrok/config.yaml")
    config_path = os.getenv("CONFIG_PATH", user_config_path)

    if not os.path.exists(config_path) and not os.path.exists("config.yaml"):
        # First run detection
        try:
            from neuralbrok.detect import detect_device
            profile = detect_device()
            print(f"NeuralBroker: no config found — detected {profile.gpu_model} · run 'neuralbrok setup' to configure")
            logger.info(f"NeuralBroker: no config found — detected {profile.gpu_model} · run 'neuralbrok setup' to configure")
        except Exception as e:
            logger.warning(f"Device detection failed: {e}")
            
        from neuralbrok.config import Config, RoutingConfig, LocalNodeConfig
        config = Config(
            local_nodes=[LocalNodeConfig(name="local", runtime="ollama", host="localhost:11434", vram_threshold=0.80)],
            routing=RoutingConfig(default_mode="cost")
        )
    else:
        # Fallback to local config.yaml if it exists (for backward compat)
        if not os.path.exists(config_path) and os.path.exists("config.yaml"):
            config_path = "config.yaml"
        elif not os.path.exists(config_path) and os.path.exists("config.yaml.example"):
            config_path = "config.yaml.example"

        try:
            config = load_config(config_path)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            print(f"NeuralBroker: Config validation failed at {config_path}")
            print(f"Error: {e}")
            print("\nPlease check your config file. Common issues:")
            print("  - YAML syntax errors (check indentation)")
            print("  - Field names: use 'runtime' not 'type', 'vram_poll_interval_seconds' not 'vram_poll_interval'")
            print("  - Missing required fields (local_nodes, routing)")
            raise SystemExit(1)

    # Start background VRAM poller (only if NOT on Vercel)
    if not os.getenv("VERCEL"):
        try:
            poller = VramPoller(gpu_id=0, poll_interval_s=config.routing.vram_poll_interval_seconds)
            await poller.start()
        except Exception as e:
            logger.warning(f"Failed to start VRAM poller: {e}")
    else:
        logger.info("Vercel Mode: GPU polling disabled.")

    # Initialize policy engine
    policy_engine = PolicyEngine(config)

    # Instantiate providers from neuralbrok.config
    for local_node in config.local_nodes:
        runtime = local_node.runtime.lower()
        name = local_node.name

        if runtime in ("ollama", "lm_studio"):
            p = OllamaProvider(name=name, host=local_node.host)
        elif runtime == "llama_cpp":
            p = LlamaCppProvider(name=name, host=local_node.host)
        else:
            logger.warning(f"Unknown runtime '{runtime}' for {name}, using Ollama")
            p = OllamaProvider(name=name, host=local_node.host)

        providers[name] = p
        provider_types[name] = "local"
        provider_costs[name] = 0.00002  # electricity cost placeholder

        # Legacy compat
        backends[name] = OllamaBackend(host=local_node.host)

    for cloud_provider in config.cloud_providers:
        api_key = os.getenv(cloud_provider.api_key_env, "")
        if not api_key:
            logger.warning(
                f"API key {cloud_provider.api_key_env} not set — "
                f"skipping {cloud_provider.name}"
            )
            continue

        name = cloud_provider.name
        base_url = cloud_provider.base_url

        cls = PROVIDER_REGISTRY.get(name)
        if cls is None:
            # Unknown provider — treat as OpenAI-compatible
            logger.warning(
                f"Unknown provider '{name}' — treating as OpenAI-compatible"
            )
            cls = GroqProvider

        # Providers with non-standard constructors
        if name == "cloudflare":
            p = cls(
                name=name,
                api_key=api_key,
                account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID", ""),
            )
        elif name == "azure_openai":
            p = cls(
                name=name,
                api_key=api_key,
                endpoint=os.getenv("AZURE_ENDPOINT", ""),
                deployment=os.getenv("AZURE_DEPLOYMENT", ""),
            )
        elif name == "bedrock":
            p = cls(name=name)
        elif name == "vertex":
            p = cls(name=name)
        elif name == "gemini":
            p = cls(name=name, api_key=api_key)
        elif name == "anthropic":
            p = cls(name=name, api_key=api_key)
        elif name == "cohere":
            p = cls(name=name, api_key=api_key)
        elif name == "replicate":
            p = cls(name=name, api_key=api_key)
        else:
            p = cls(name=name, base_url=base_url, api_key=api_key)

        providers[name] = p
        provider_types[name] = "cloud"
        provider_costs[name] = cloud_provider.cost_per_1k_tokens

        # Legacy compat
        backends[name] = GroqBackend(base_url=base_url, api_key=api_key)

    # Give the policy engine access to provider objects for model-support scoring
    policy_engine.set_providers(providers)

    logger.info(f"Loaded {len(providers)} provider(s): {list(providers.keys())}")
    yield

    if poller:
        await poller.stop()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="NeuralBroker", version="0.4.0", lifespan=lifespan)


# ── Auth Helper ───────────────────────────────────────────────────────────────

def _check_auth(request: Request) -> None:
    """Check bearer token against NB_API_KEY env var.

    If NB_API_KEY is not set, auth is disabled (development mode).
    """
    expected = os.getenv("NB_API_KEY", "")
    if not expected:
        return  # Auth disabled

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = auth[7:]
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _openai_error(status: int, message: str, error_type: str = "server_error") -> dict:
    """Create an OpenAI-format error response body."""
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": status,
        }
    }


# ── OpenAI Proxy Endpoints ───────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completion endpoint.

    Routes to local or cloud backend based on VRAM state and policy mode.
    Supports both streaming and non-streaming responses.
    """
    _check_auth(request)

    try:
        body = await request.json()
    except Exception:
        return Response(
            content=json.dumps(_openai_error(400, "Invalid JSON body", "invalid_request_error")),
            status_code=400,
            media_type="application/json",
        )

    stream = body.get("stream", False)

    # Get current VRAM snapshot (cached, <1ms)
    vram_snapshot = poller.latest()

    # Update VRAM gauge metric
    if vram_snapshot:
        total = vram_snapshot.vram_used_gb + vram_snapshot.vram_free_gb
        if total > 0:
            nb_metrics.set_vram_utilization(
                vram_snapshot.gpu_id, vram_snapshot.vram_used_gb / total
            )

    # Route decision
    decision = await policy_engine.decide_async(
        request_body=body,
        vram=vram_snapshot,
        available_providers=list(providers.keys()),
        provider_types=provider_types,
        provider_costs=provider_costs,
        requested_model=body.get("model", ""),
    )

    nb_metrics.record_routing_latency(decision.latency_ms)

    if not decision.backend_chosen:
        return Response(
            content=json.dumps(_openai_error(503, "No available providers")),
            status_code=503,
            media_type="application/json",
        )

    # Try the chosen backend, then fallback chain
    backends_to_try = [decision.backend_chosen] + decision.fallback_chain
    last_error = None

    for backend_name in backends_to_try:
        if backend_name not in providers:
            continue

        provider = providers[backend_name]

        try:
            if stream:
                # Streaming response
                async def stream_generator(prov=provider, bname=backend_name):
                    try:
                        start_req = time.perf_counter()
                        first = True
                        async for chunk in prov.chat(body, stream=True):
                            if first:
                                first = False
                            yield chunk
                        if policy_engine:
                            policy_engine.record_latency(bname, (time.perf_counter() - start_req) * 1000)
                        policy_engine.record_success(bname)
                        nb_metrics.record_request(bname, decision.policy_mode.value, "ok")
                    except ProviderError as e:
                        policy_engine.record_error(bname)
                        nb_metrics.record_provider_error(bname)
                        nb_metrics.record_request(bname, decision.policy_mode.value, "error")
                        logger.error(f"Streaming error from {bname}: {e}")
                        error_chunk = {
                            "id": "error",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": body.get("model", ""),
                            "choices": [{
                                "index": 0,
                                "delta": {"content": f"\n\n[NeuralBroker: {bname} error — {e}]"},
                                "finish_reason": "stop",
                            }],
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        policy_engine.record_error(bname)
                        nb_metrics.record_provider_error(bname)
                        logger.error(f"Unexpected streaming error from {bname}: {e}")

                vram_pct = "0%"
                if vram_snapshot:
                    total = vram_snapshot.vram_used_gb + vram_snapshot.vram_free_gb
                    if total > 0:
                        vram_pct = f"{int(vram_snapshot.vram_used_gb / total * 100)}%"

                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-NB-Backend": backend_name,
                        "X-NB-VRAM": vram_pct,
                        "X-NB-Cost": f"${provider_costs.get(backend_name, 0):.6f}",
                        "X-NB-RoutingMode": decision.policy_mode.value,
                        # Legacy headers
                        "X-Route-Decision": backend_name,
                        "X-Vram-Used-Gb": str(round(vram_snapshot.vram_used_gb, 2)),
                        "X-Vram-Free-Gb": str(round(vram_snapshot.vram_free_gb, 2)),
                    },
                )
            else:
                # Non-streaming response
                result_text = ""
                start_req = time.perf_counter()
                async for chunk in provider.chat(body, stream=False):
                    result_text += chunk

                if policy_engine:
                    policy_engine.record_latency(backend_name, (time.perf_counter() - start_req) * 1000)
                policy_engine.record_success(backend_name)
                nb_metrics.record_request(backend_name, decision.policy_mode.value, "ok")

                vram_pct = "0%"
                if vram_snapshot:
                    total = vram_snapshot.vram_used_gb + vram_snapshot.vram_free_gb
                    if total > 0:
                        vram_pct = f"{int(vram_snapshot.vram_used_gb / total * 100)}%"

                return Response(
                    content=result_text,
                    media_type="application/json",
                    headers={
                        "X-NB-Backend": backend_name,
                        "X-NB-VRAM": vram_pct,
                        "X-NB-Cost": f"${provider_costs.get(backend_name, 0):.6f}",
                        "X-NB-RoutingMode": decision.policy_mode.value,
                        "X-NB-Classified": decision.classified_as or "none",
                    },
                )

        except ProviderError as e:
            policy_engine.record_error(backend_name)
            nb_metrics.record_provider_error(backend_name)
            nb_metrics.record_request(backend_name, decision.policy_mode.value, "error")
            last_error = e
            logger.error(f"Provider {backend_name} failed: {e}")
            continue

        except Exception as e:
            policy_engine.record_error(backend_name)
            nb_metrics.record_provider_error(backend_name)
            last_error = e
            logger.error(f"Unexpected error from {backend_name}: {e}")
            continue

    # All providers exhausted
    tried = [b for b in backends_to_try if b in providers]
    return Response(
        content=json.dumps(_openai_error(
            503,
            f"All providers failed. Tried: {', '.join(tried)}. "
            f"Last error: {last_error}"
        )),
        status_code=503,
        media_type="application/json",
        headers={"X-NB-Backends-Tried": ",".join(tried)},
    )


@app.get("/v1/models")
async def list_models(request: Request):
    """List available models from all configured providers."""
    _check_auth(request)
    models = []
    for name, provider in providers.items():
        models.append({
            "id": name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": f"neuralbroker/{provider.provider_type}",
        })
    return {"object": "list", "data": models}


# ── Internal NB Endpoints ────────────────────────────────────────────────────

@app.get("/nb/vram")
async def nb_vram():
    """Current VRAM snapshot per GPU."""
    if poller is None:
        return Response(
            content=json.dumps(_openai_error(503, "VRAM poller not initialized")),
            status_code=503,
            media_type="application/json",
        )
    snap = poller.latest()
    total = snap.vram_used_gb + snap.vram_free_gb
    return {
        f"gpu_{snap.gpu_id}": {
            "used_gb": round(snap.vram_used_gb, 3),
            "free_gb": round(snap.vram_free_gb, 3),
            "total_gb": round(total, 3),
            "utilization": round(snap.vram_used_gb / total, 3) if total > 0 else 0,
            "available": True,
        }
    }


@app.get("/nb/routing-log")
async def nb_routing_log():
    """Last 500 routing decisions."""
    if policy_engine is None:
        return {"decisions": []}
    return {"decisions": policy_engine.get_routing_log()}


@app.get("/nb/providers")
async def nb_providers():
    """List configured providers with health and circuit breaker status."""
    if policy_engine is None:
        return {"providers": []}
    statuses = policy_engine.get_provider_statuses(
        list(providers.keys()), provider_types
    )
    # Augment with SUPPORTED_MODELS count and api_key_configured flag
    for entry in statuses:
        prov = providers.get(entry["name"])
        entry["supported_model_count"] = (
            len(prov.SUPPORTED_MODELS)
            if prov and hasattr(prov, "SUPPORTED_MODELS")
            else 0
        )
        entry["api_key_configured"] = True  # Only configured providers are registered
    return {"providers": statuses}


@app.get("/nb/latency")
async def nb_latency():
    if policy_engine:
        return policy_engine.get_latency_stats()
    return {}

@app.get("/nb/stats")
async def nb_stats():
    """Aggregate statistics: requests, local %, cloud %, cost saved."""
    if policy_engine is None:
        return {}
    return policy_engine.get_stats()


@app.post("/nb/mode")
async def nb_set_mode(body: dict):
    """Change routing mode at runtime (cost | speed | fallback | smart)."""
    if policy_engine is None:
        return {"error": "policy engine not initialized"}
    mode = body.get("mode", "cost")
    try:
        policy_engine.set_mode(mode)
        return {"mode": policy_engine.mode.value}
    except (ValueError, KeyError) as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")


# ── Observability ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "backends": list(providers.keys()),
        "mode": policy_engine.mode.value if policy_engine else "unknown",
    }


@app.get("/nb/recommend")
async def nb_recommend(workload: str = "chat,coding,reasoning"):
    """
    Live model recommendations for your hardware.
    Returns local models that fit in VRAM + Ollama Cloud options.
    
    Query params:
        workload: comma-separated capabilities (chat, coding, math, vision, reasoning, rag, tools)
    """
    from neuralbrok.detect import detect_device
    from neuralbrok.ollama_catalog import (
        fetch_latest_ollama_models, assess_hardware,
        get_cloud_recommendations, get_runnable_local_models,
    )
    from neuralbrok.models import get_runnable_models, FALLBACK_MODELS, get_tok_per_sec

    wl = [w.strip() for w in workload.split(",") if w.strip()]
    profile = detect_device()
    hw = assess_hardware(profile.vram_gb, getattr(profile, "bandwidth_gbps", None))

    # Local models
    runnable = get_runnable_models(profile.vram_gb, profile.ram_gb, profile.gpu_model)
    bw = getattr(profile, "bandwidth_gbps", None)
    local_picks = []
    for m in runnable[:6]:
        weight = m.weight_gb if m.weight_gb > 0 else m.vram_gb
        tps = (bw / (weight + 1.0)) if bw else get_tok_per_sec(m, profile.gpu_model)
        local_picks.append({
            "tag": m.name,
            "params_b": m.params_b,
            "vram_gb": m.vram_gb,
            "est_tok_per_sec": round(tps, 1),
            "capabilities": m.capabilities,
            "is_installed": m.is_installed,
            "run_cmd": f"ollama run {m.ollama_tag}",
        })

    # Ollama Cloud options
    cloud_picks = []
    if hw["suggest_cloud"] or not local_picks:
        for cm in get_cloud_recommendations(profile.vram_gb, wl)[:4]:
            cloud_picks.append({
                "tag": cm["tag"],
                "name": cm["name"],
                "description": cm["description"],
                "capabilities": cm["capabilities"],
                "tier": cm["tier"],
                "run_cmd": f"ollama run {cm['tag']}",
                "api_model": cm["tag"],
            })

    # Live catalog from Ollama registry
    try:
        live = fetch_latest_ollama_models(timeout=3.0)
        live_runnable = get_runnable_local_models(profile.vram_gb, live)
        trending = [
            {"tag": m.tag, "params_b": m.params_b, "vram_gb": round(m.vram_gb, 1),
             "description": m.description[:80], "capabilities": m.capabilities}
            for m in live_runnable[:5]
        ]
    except Exception:
        trending = []

    return {
        "hardware": {
            "gpu": profile.gpu_model,
            "vram_gb": round(profile.vram_gb, 1),
            "bandwidth_gbps": getattr(profile, "bandwidth_gbps", None),
            "tier": hw["tier"],
            "assessment": hw["message"],
        },
        "workload": wl,
        "local_models": local_picks,
        "ollama_cloud": cloud_picks,
        "trending_local": trending,
        "suggest_cloud": hw["suggest_cloud"],
    }


@app.get("/nb/hardware")
async def nb_hardware():
    """Detailed hardware profile from whatmodels logic."""
    from neuralbrok.detect import detect_device
    profile = detect_device()
    return {
        "model": profile.gpu_model,
        "vendor": profile.gpu_vendor,
        "vram_gb": round(profile.vram_gb, 1),
        "bandwidth_gbps": profile.bandwidth_gbps,
        "platform": profile.platform
    }

@app.get("/telemetry")
async def telemetry():
    """Live VRAM snapshot from background poller. For debugging."""
    if poller is None:
        return {"gpu_id": 0, "vram_used_gb": 0.0, "vram_free_gb": 0.0, "timestamp": None}
    snap = poller.latest()
    return {
        "gpu_id": snap.gpu_id,
        "vram_used_gb": round(snap.vram_used_gb, 3),
        "vram_free_gb": round(snap.vram_free_gb, 3),
        "timestamp": snap.timestamp.isoformat() if snap.timestamp else None,
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    text, content_type = nb_metrics.get_metrics_response()
    return Response(content=text, media_type=content_type)


# ── Dashboard & Onboarding ───────────────────────────────────────────────────

@app.get("/")
async def root_redirect():
    """Redirect to onboarding/dashboard."""
    stats = policy_engine.get_stats() if policy_engine else {}
    log = policy_engine.get_routing_log() if policy_engine else []
    if stats.get("total_requests", 0) == 0 and len(log) == 0:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/onboarding")
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")

@app.get("/onboarding")
async def onboarding_page():
    """Serve the interactive first-run onboarding page."""
    stats = policy_engine.get_stats()
    log = policy_engine.get_routing_log()
    if stats.get("total_requests", 0) > 0 or len(log) > 0:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/dashboard")
        
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NeuralBroker Onboarding</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root { --bg: #0e0d0b; --ink-1: #ffffff; --ink-2: #a1a1aa; --ink-3: #71717a; --ink-4: #3f3f46; --accent: oklch(0.78 0.14 65); --accent-dim: color-mix(in srgb, var(--accent) 20%, transparent); --success: #4ade80; --border: #27272a; --panel: #18181b; }
  body { background: var(--bg); color: var(--ink-1); font-family: 'IBM Plex Sans', sans-serif; margin: 0; padding: 40px; line-height: 1.5; }
  .container { max-width: 600px; margin: 0 auto; }
  h1 { font-weight: 500; margin-bottom: 40px; display: flex; align-items: center; gap: 12px; }
  .logo { color: var(--accent); }
  .timeline { position: relative; padding-left: 30px; margin-bottom: 40px; }
  .timeline::before { content: ''; position: absolute; left: 8px; top: 10px; bottom: -20px; width: 2px; background: var(--border); z-index: 0; }
  .step { position: relative; margin-bottom: 40px; z-index: 1; }
  .step-icon { position: absolute; left: -30px; top: 2px; width: 18px; height: 18px; border-radius: 50%; background: var(--bg); border: 2px solid var(--border); display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; }
  .step.active .step-icon { border-color: var(--accent); color: var(--accent); }
  .step.done .step-icon { border-color: var(--success); color: var(--success); }
  .step.done .step-icon::after { content: '✓'; }
  .step-title { font-weight: 500; margin-bottom: 8px; font-size: 1.1rem; }
  .step.active .step-title { color: var(--ink-1); }
  .step.done .step-title { color: var(--success); }
  .step-content { color: var(--ink-2); font-size: 0.95rem; }
  .mono { font-family: 'JetBrains Mono', monospace; font-size: 0.85em; }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px; margin-top: 12px; }
  button { background: var(--accent-dim); color: var(--accent); border: 1px solid var(--accent); padding: 6px 12px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; cursor: pointer; margin-top: 12px; transition: 0.2s; }
  button:hover { background: var(--accent); color: var(--bg); }
  .banner { display: none; background: color-mix(in srgb, var(--success) 20%, transparent); color: var(--success); border: 1px solid var(--success); padding: 16px; border-radius: 6px; text-align: center; margin-top: 40px; font-weight: 500; }
  #test-response { display: none; margin-top: 16px; border-color: var(--success); }
  .meta { color: var(--ink-3); margin-top: 8px; font-size: 0.85rem; }
</style>
</head>
<body>
  <div class="container">
  <h1><span class="logo">●</span> NeuralBroker Onboarding</h1>
  
  <div id="hw-profile" class="panel" style="margin-bottom: 30px; display: none;">
    <div style="color: var(--ink-3); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 4px;">Detected Hardware</div>
    <div style="display: flex; justify-content: space-between; align-items: center;">
      <div id="hw-name" style="font-weight: 600; color: var(--accent);">Detecting...</div>
      <div id="hw-bandwidth" class="mono" style="font-size: 0.9rem;">-- GB/s</div>
    </div>
  </div>

  <div class="timeline" id="timeline">
    
    <div class="step" id="step-config">
      <div class="step-icon"></div>
      <div class="step-title" id="title-config">Config</div>
      <div class="step-content mono" id="content-config">Checking...</div>
    </div>
    
    <div class="step" id="step-ollama">
      <div class="step-icon"></div>
      <div class="step-title" id="title-ollama">Local runtime</div>
      <div class="step-content mono" id="content-ollama">Checking...</div>
    </div>
    
    <div class="step" id="step-cloud">
      <div class="step-icon"></div>
      <div class="step-title" id="title-cloud">Cloud key</div>
      <div class="step-content mono" id="content-cloud">Checking...</div>
    </div>
    
    <div class="step" id="step-request">
      <div class="step-icon"></div>
      <div class="step-title" id="title-request">First request</div>
      <div class="step-content mono" id="content-request">
        Waiting for previous steps...
      </div>
    </div>
    
  </div>
  
  <div class="banner" id="success-banner">
    ✓ NeuralBroker is routing. Redirecting to dashboard...
  </div>
</div>

<script>
  let isDone = false;
  
  async function poll() {
    if (isDone) return;
    try {
      const hRes = await fetch('/health');
      const hData = await hRes.json();
      
      const sRes = await fetch('/nb/stats');
      const sData = await sRes.json();
      
      const pRes = await fetch('/nb/providers');
      const pData = await pRes.json();
      
      const hwRes = await fetch('/nb/hardware');
      const hwData = await hwRes.json();
      
      updateUI(hData, sData, pData, hwData);
    } catch (e) { console.error(e); }
  }
  
  function updateUI(health, stats, providers, hardware) {
    if (hardware && hardware.model) {
      document.getElementById('hw-profile').style.display = 'block';
      document.getElementById('hw-name').innerText = hardware.model;
      document.getElementById('hw-bandwidth').innerText = (hardware.bandwidth_gbps ? hardware.bandwidth_gbps + ' GB/s' : 'Bandwidth unknown');
    }

    // 1. Config
    const sc = document.getElementById('step-config');
    sc.className = 'step done';
    document.getElementById('title-config').innerText = '✓ Config loaded';
    document.getElementById('content-config').innerHTML = `~/.neuralbrok/config.yaml · ${health.mode}-mode`;
    
    // 2. Ollama
    const so = document.getElementById('step-ollama');
    const localProvs = providers.providers.filter(p => p.type === 'local');
    const isLocalOk = localProvs.some(p => p.healthy);
    
    if (isLocalOk) {
      so.className = 'step done';
      document.getElementById('title-ollama').innerText = '✓ Local runtime connected';
      document.getElementById('content-ollama').innerHTML = `Models available`;
    } else {
      so.className = 'step active';
      document.getElementById('title-ollama').innerText = '◐ Waiting for Ollama...';
      document.getElementById('content-ollama').innerHTML = `
        Start Ollama: open a terminal and run <span style="color:var(--ink-1)">ollama serve</span><br>
        Then pull a model: <span style="color:var(--ink-1)">ollama pull qwen3:8b</span><br>
        <button onclick="poll()">Check again</button>
      `;
      return; // block next
    }
    
    // 3. Cloud
    const sq = document.getElementById('step-cloud');
    const cloudProvs = providers.providers.filter(p => p.type === 'cloud' && p.healthy);
    if (cloudProvs.length > 0) {
      sq.className = 'step done';
      document.getElementById('title-cloud').innerText = '✓ Cloud provider ready';
      document.getElementById('content-cloud').innerHTML = `${cloudProvs[0].name} configured`;
    } else {
      sq.className = 'step active';
      document.getElementById('title-cloud').innerText = '◐ Add a cloud key (needed for VRAM spillover)';
      document.getElementById('content-cloud').innerHTML = `
        Add to your .env file:<br>
        <span style="color:var(--ink-1)">GROQ_KEY=...</span>   ← get free key at console.groq.com<br>
        Then restart NeuralBroker.<br>
        <button onclick="poll()">I've added a key — check again</button>
      `;
      return;
    }
    
    // 4. Request
    const sr = document.getElementById('step-request');
    if (stats.total_requests > 0) {
      sr.className = 'step done';
      document.getElementById('title-request').innerText = '✓ Request routed successfully';
      isDone = true;
      document.getElementById('success-banner').style.display = 'block';
      setTimeout(() => { window.location.href = '/dashboard'; }, 2000);
    } else {
      sr.className = 'step active';
      document.getElementById('title-request').innerText = '◐ Send your first request';
      if (!document.getElementById('btn-test')) {
          document.getElementById('content-request').innerHTML = `
            Change one line in your code:<br><br>
            <div class="panel">
            client = OpenAI(<br>
            &nbsp;&nbsp;base_url="http://localhost:8000/v1",<br>
            &nbsp;&nbsp;api_key="nb_..."<br>
            )
            </div><br>
            Or test with curl:<br>
            <div class="panel">
            $ curl http://localhost:8000/v1/chat/completions \\<br>
            &nbsp;&nbsp;-H "Content-Type: application/json" \\<br>
            &nbsp;&nbsp;-d '{"model":"{resolve_model("default")}","messages":[{"role":"user","content":"hello"}]}'
            </div>
            <button id="btn-test" onclick="fireTest()">Send test request</button>
            <div id="test-response" class="panel"></div>
            <div id="test-meta" class="meta"></div>
          `;
      }
    }
  }
  
  async function fireTest() {
    const btn = document.getElementById('btn-test');
    btn.innerText = 'Sending...';
    try {
      const res = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: '{resolve_model("default")}', messages: [{role: 'user', content: 'reply with exactly three words'}], stream: false })
      });
      const data = await res.json();
      const be = res.headers.get('X-NB-Backend');
      const cost = res.headers.get('X-NB-Cost');
      
      const out = document.getElementById('test-response');
      out.style.display = 'block';
      out.innerText = JSON.stringify(data, null, 2);
      
      document.getElementById('test-meta').innerHTML = `Routed to: <span style="color:var(--success)">${be}</span> · ${cost}`;
      
      // The poll will catch the incremented stats and transition
      setTimeout(poll, 500);
    } catch (e) {
      btn.innerText = 'Failed';
      console.error(e);
    }
  }
  
  setInterval(poll, 2000);
  poll();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)

# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.get("/dashboard")
async def dashboard_page():
    """Serve the local-first dashboard (embedded)."""
    return HTMLResponse(content=DASHBOARD_HTML)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


DASHBOARD_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>NeuralBroker — Live Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0a;
    --bg-2: #121212;
    --bg-3: #1a1a1a;
    --line: #222222;
    --ink: #ffffff;
    --ink-2: #e0e0e0;
    --ink-3: #999999;
    --ink-4: #444444;
    --accent: #ff87ff; /* CLI Pink */
    --ok: #5fff00;    /* Matrix Green */
    --hot: #ff5555;
    --cool: #00ffff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    background: var(--bg); color: var(--ink);
    font-size: 14px; line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }
  .mono { font-family: 'JetBrains Mono', monospace; }

  /* Layout */
  .header {
    display: flex; align-items: center; gap: 16px;
    padding: 16px 24px; border-bottom: 1px solid var(--line);
    background: var(--bg-2);
  }
  .header .logo {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600; font-size: 14px;
  }
  .header .mode {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: var(--accent); border: 1px solid var(--accent);
    padding: 3px 10px; border-radius: 20px;
    background: color-mix(in oklab, var(--accent) 10%, transparent);
  }
  .header .status {
    margin-left: auto; display: flex; align-items: center; gap: 8px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--ink-3);
  }
  .header .status .led {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--ok); box-shadow: 0 0 6px var(--ok);
  }

  .grid {
    display: grid; grid-template-columns: 280px 1fr 320px;
    gap: 0; min-height: calc(100vh - 53px);
  }

  .panel {
    border-right: 1px solid var(--line);
    display: flex; flex-direction: column;
  }
  .panel:last-child { border-right: none; }
  .panel-head {
    padding: 14px 18px; border-bottom: 1px solid var(--line);
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: var(--ink-3); text-transform: uppercase; letter-spacing: 0.1em;
    background: var(--bg-2);
  }
  .panel-body { padding: 18px; flex: 1; overflow-y: auto; }

  /* VRAM Gauge */
  .gauge-wrap { display: flex; flex-direction: column; align-items: center; gap: 16px; padding-top: 20px; }
  .gauge-svg { width: 180px; height: 180px; }
  .gauge-ring { fill: none; stroke-width: 12; stroke-linecap: round; }
  .gauge-bg { stroke: var(--line); }
  .gauge-fg { transition: stroke-dashoffset 0.6s ease, stroke 0.3s ease; }
  .gauge-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 36px; font-weight: 600;
    fill: var(--ink);
  }
  .gauge-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; fill: var(--ink-3);
    text-transform: uppercase; letter-spacing: 0.1em;
  }

  .stats-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 24px; width: 100%;
  }
  .stat-card {
    background: var(--bg-2); border: 1px solid var(--line);
    border-radius: 8px; padding: 12px;
  }
  .stat-card .label {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    color: var(--ink-3); text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 4px;
  }
  .stat-card .value {
    font-family: 'JetBrains Mono', monospace; font-size: 20px;
    font-weight: 500; color: var(--ink);
  }
  .stat-card .value.accent { color: var(--accent); }

  /* Routing Feed */
  .feed { display: flex; flex-direction: column; gap: 0; }
  .feed-item {
    padding: 10px 18px; border-bottom: 1px solid var(--line);
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    display: grid; grid-template-columns: 1fr auto; gap: 8px;
    transition: background 0.2s ease;
  }
  .feed-item:hover { background: var(--bg-2); }
  .feed-item .backend { color: var(--ink); font-weight: 500; }
  .feed-item .backend.local { color: var(--accent); }
  .feed-item .backend.cloud { color: var(--cool); }
  .feed-item .meta { color: var(--ink-3); font-size: 11px; }
  .feed-item .reason {
    font-size: 10px; color: var(--ink-4);
    padding: 1px 6px; border: 1px solid var(--line);
    border-radius: 3px; align-self: start;
  }
  .feed-empty {
    padding: 40px 18px; text-align: center;
    color: var(--ink-4); font-family: 'JetBrains Mono', monospace; font-size: 12px;
  }

  /* Cost Chart */
  .chart-wrap { position: relative; width: 100%; height: 240px; }
  .chart-canvas { width: 100%; height: 100%; }
  .chart-legend {
    display: flex; gap: 16px; margin-top: 12px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
  }
  .chart-legend .dot {
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block; margin-right: 6px;
  }
  .chart-legend .local .dot { background: var(--accent); }
  .chart-legend .cloud .dot { background: var(--cool); }

  .providers-list { margin-top: 20px; }
  .prov-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid var(--line);
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
  }
  .prov-item .prov-led {
    width: 6px; height: 6px; border-radius: 50%;
  }
  .prov-item .prov-led.ok { background: var(--ok); box-shadow: 0 0 4px var(--ok); }
  .prov-item .prov-led.fail { background: var(--hot); box-shadow: 0 0 4px var(--hot); }
  .prov-item .prov-name { color: var(--ink); font-weight: 500; }
  .prov-item .prov-type { color: var(--ink-4); font-size: 10px; }

  /* Recommendations */
  .rec-card {
    background: var(--bg-2); border: 1px solid var(--line);
    border-radius: 8px; padding: 12px; margin-bottom: 12px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
  }
  .rec-card.cloud { border-left: 4px solid var(--cool); }
  .rec-card.local { border-left: 4px solid var(--accent); }
  .rec-title { font-weight: 600; font-size: 13px; color: var(--ink); margin-bottom: 4px; display: flex; justify-content: space-between; }
  .rec-desc { color: var(--ink-3); line-height: 1.4; margin-bottom: 8px; }
  .rec-meta { display: flex; gap: 10px; color: var(--ink-4); font-size: 10px; }
  .rec-badge { padding: 1px 6px; border-radius: 3px; background: var(--bg-3); color: var(--ink-2); }
  
  .hw-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 100px;
    background: var(--bg-2); border: 1px solid var(--line);
    font-size: 11px; color: var(--ink-2);
    margin-bottom: 16px;
  }
  .hw-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--ok); box-shadow: 0 0 6px var(--ok); }
  .hw-tier { color: var(--accent); font-weight: 600; text-shadow: 0 0 8px color-mix(in srgb, var(--accent) 50%, transparent); }

  /* Mode switcher */
  .mode-switcher { display: flex; gap: 4px; margin-left: 12px; }
  .mode-btn {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    padding: 3px 9px; border-radius: 12px; cursor: pointer;
    border: 1px solid var(--line); background: transparent; color: var(--ink-3);
    transition: 0.15s;
  }
  .mode-btn:hover { border-color: var(--accent); color: var(--accent); }
  .mode-btn.active { border-color: var(--accent); color: var(--accent); background: color-mix(in oklab, var(--accent) 12%, transparent); }

  /* Waterfall bar in feed */
  .feed-item { flex-direction: column; gap: 4px; }
  .feed-row { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: start; }
  .waterfall-bar-wrap { height: 3px; background: var(--line); border-radius: 2px; margin-top: 2px; width: 100%; }
  .waterfall-bar { height: 3px; border-radius: 2px; transition: width 0.3s ease; min-width: 2px; }
  .waterfall-bar.local { background: var(--accent); }
  .waterfall-bar.cloud { background: var(--cool); }

  /* Per-provider cost bars */
  .prov-cost-bar-wrap { flex: 1; height: 4px; background: var(--line); border-radius: 2px; margin: 0 8px; }
  .prov-cost-bar { height: 4px; border-radius: 2px; background: var(--accent); }
  .prov-cost-bar.cloud { background: var(--cool); }
  .prov-item { align-items: center; }
  .prov-cost-label { font-size: 10px; color: var(--ink-4); min-width: 52px; text-align: right; }

  @media (max-width: 1200px) {
    .grid { grid-template-columns: 280px 1fr; }
    .panel:last-child { display: none; } /* Hide chart on smaller screens */
  }
  @media (max-width: 900px) {
    .grid { grid-template-columns: 1fr; }
    .panel { border-right: none; border-bottom: 1px solid var(--line); }
  }
</style>
</head>
<body>

<div class="header">
  <span class="logo">neuralbroker</span>
  <span class="mode" id="mode-badge">cost-mode</span>
  <div class="mode-switcher">
    <button class="mode-btn active" id="btn-cost" onclick="setMode('cost')">cost</button>
    <button class="mode-btn" id="btn-speed" onclick="setMode('speed')">speed</button>
    <button class="mode-btn" id="btn-fallback" onclick="setMode('fallback')">fallback</button>
    <button class="mode-btn" id="btn-smart" onclick="setMode('smart')">smart</button>
  </div>
  <div class="status">
    <span class="led"></span>
    <span id="status-text">polling · 500ms</span>
  </div>
</div>

<div class="grid">
  <!-- VRAM Gauge Panel -->
  <div class="panel">
    <div class="panel-head">VRAM Utilization</div>
    <div class="panel-body">
      <div class="gauge-wrap">
        <svg class="gauge-svg" viewBox="0 0 200 200">
          <circle class="gauge-ring gauge-bg" cx="100" cy="100" r="80" transform="rotate(-90 100 100)" stroke-dasharray="502.65" />
          <circle class="gauge-ring gauge-fg" id="gauge-arc" cx="100" cy="100" r="80" transform="rotate(-90 100 100)" stroke-dasharray="502.65" stroke-dashoffset="502.65" stroke="var(--accent)" />
          <text class="gauge-label" id="gauge-pct" x="100" y="100" text-anchor="middle" dominant-baseline="central">0%</text>
          <text class="gauge-sub" x="100" y="130" text-anchor="middle">VRAM</text>
        </svg>

        <div class="stats-grid">
          <div class="stat-card">
            <div class="label">Total Requests</div>
            <div class="value" id="stat-total">0</div>
          </div>
          <div class="stat-card">
            <div class="label">Local %</div>
            <div class="value accent" id="stat-local">0%</div>
          </div>
          <div class="stat-card">
            <div class="label">Saved</div>
            <div class="value accent" id="stat-saved">$0.00</div>
          </div>
          <div class="stat-card">
            <div class="label">Cloud Cost</div>
            <div class="value" id="stat-cloud">$0.00</div>
          </div>
        </div>

        <div class="providers-list" id="providers-list"></div>
      </div>
    </div>
  </div>

  <!-- Routing Feed Panel -->
  <div class="panel">
    <div class="panel-head">Routing Feed — last 20 decisions</div>
    <div class="panel-body" style="padding:0;">
      <div class="feed" id="feed">
        <div class="feed-empty">Waiting for routing decisions…</div>
      </div>
    </div>
  </div>

  <!-- Cost Chart Panel -->
  <div class="panel">
    <div class="panel-head">Hardware & Recommendations</div>
    <div class="panel-body">
      <div id="hw-info">
        <div class="hw-badge">
          <span class="dot"></span>
          <span id="hw-model">Detecting hardware...</span>
        </div>
        <div id="hw-assessment" style="font-size: 12px; color: var(--ink-3); margin-bottom: 20px; line-height: 1.5;"></div>
      </div>

      <div class="panel-head" style="padding-left:0; background:transparent; border-bottom:none; margin-bottom:10px;">Recommended Models</div>
      <div id="recs-list">
        <div class="feed-empty">Analyzing workload...</div>
      </div>
      
      <div style="margin-top:20px;">
        <div class="chart-wrap" style="height: 140px;">
          <canvas class="chart-canvas" id="cost-chart"></canvas>
        </div>
        <div class="chart-legend">
          <span class="local"><span class="dot"></span>Local</span>
          <span class="cloud"><span class="dot"></span>Cloud</span>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
(function() {
  const POLL_MS = 500;
  const MAX_FEED = 20;
  const MAX_CHART_POINTS = 60;

  // Chart data
  const chartLocal = [];
  const chartCloud = [];

  // ── VRAM Gauge ──
  const gaugeArc = document.getElementById('gauge-arc');
  const gaugePct = document.getElementById('gauge-pct');
  const CIRCUMFERENCE = 2 * Math.PI * 80;

  function updateGauge(util) {
    const pct = Math.round(util * 100);
    const offset = CIRCUMFERENCE * (1 - util);
    gaugeArc.style.strokeDashoffset = offset;
    gaugePct.textContent = pct + '%';

    // Color based on utilization
    if (util > 0.9) gaugeArc.style.stroke = 'var(--hot)';
    else if (util > 0.7) gaugeArc.style.stroke = 'var(--accent)';
    else gaugeArc.style.stroke = 'var(--ok)';
  }

  // ── Fetch VRAM ──
  async function fetchVram() {
    try {
      const r = await fetch('/nb/vram');
      const data = await r.json();
      const gpu0 = data.gpu_0;
      if (gpu0) updateGauge(gpu0.utilization);
    } catch(e) {}
  }

  // ── Mode switcher ──
  async function setMode(mode) {
    try {
      const r = await fetch('/nb/mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: mode }),
      });
      const data = await r.json();
      updateModeBadge(data.mode || mode);
    } catch(e) {}
  }

  function updateModeBadge(mode) {
    document.getElementById('mode-badge').textContent = mode + '-mode';
    ['cost', 'speed', 'fallback', 'smart'].forEach(function(m) {
      var btn = document.getElementById('btn-' + m);
      if (btn) btn.className = 'mode-btn' + (m === mode ? ' active' : '');
    });
  }

  // ── Fetch Stats ──
  async function fetchStats() {
    try {
      const r = await fetch('/nb/stats');
      const s = await r.json();
      document.getElementById('stat-total').textContent = s.total_requests.toLocaleString();
      document.getElementById('stat-local').textContent = s.local_pct + '%';
      document.getElementById('stat-saved').textContent = '$' + s.total_saved.toFixed(2);
      document.getElementById('stat-cloud').textContent = '$' + s.total_cost_cloud.toFixed(2);

      if (s.routing_mode) updateModeBadge(s.routing_mode);

      // Chart data
      chartLocal.push(s.total_cost_local);
      chartCloud.push(s.total_cost_cloud);
      if (chartLocal.length > MAX_CHART_POINTS) { chartLocal.shift(); chartCloud.shift(); }
      drawChart();

      // Per-provider cost bars
      if (s.provider_costs) drawProviderCosts(s.provider_costs);
    } catch(e) {}
  }

  // ── Fetch Routing Log ──
  async function fetchFeed() {
    try {
      const r = await fetch('/nb/routing-log');
      const data = await r.json();
      const decisions = (data.decisions || []).slice(-MAX_FEED).reverse();

      const feed = document.getElementById('feed');
      if (decisions.length === 0) {
        feed.innerHTML = '<div class="feed-empty">Waiting for routing decisions…</div>';
        return;
      }

      feed.innerHTML = decisions.map(function(d) {
        const cls = d.backend && d.backend.includes('local') ? 'local' : 'cloud';
        const barPct = Math.min(100, Math.round((d.latency_ms || 0) / 200 * 100));
        return '<div class="feed-item">' +
          '<div class="feed-row">' +
            '<div>' +
              '<span class="backend ' + cls + '">' + (d.backend || '—') + '</span>' +
              '<div class="meta">' + d.mode + ' · ' + d.latency_ms + 'ms · vram ' + Math.round((d.vram_util || 0) * 100) + '%</div>' +
            '</div>' +
            '<span class="reason">' + (d.reason || '') + '</span>' +
          '</div>' +
          '<div class="waterfall-bar-wrap"><div class="waterfall-bar ' + cls + '" style="width:' + barPct + '%"></div></div>' +
        '</div>';
      }).join('');
    } catch(e) {}
  }

  // ── Fetch Providers ──
  let _providerCosts = {};
  async function fetchProviders() {
    try {
      const r = await fetch('/nb/providers');
      const data = await r.json();
      const list = document.getElementById('providers-list');

      list.innerHTML = (data.providers || []).map(function(p) {
        const ledCls = p.healthy ? 'ok' : 'fail';
        const cost = _providerCosts[p.name] || 0;
        const costStr = cost > 0 ? '$' + cost.toFixed(4) : '';
        return '<div class="prov-item">' +
          '<span class="prov-led ' + ledCls + '"></span>' +
          '<span class="prov-name">' + p.name + '</span>' +
          '<span class="prov-type">' + p.type + '</span>' +
          (costStr ? '<span class="prov-cost-label">' + costStr + '</span>' : '') +
        '</div>';
      }).join('');
    } catch(e) {}
  }

  // ── Per-provider cost bar chart ──
  function drawProviderCosts(costs) {
    _providerCosts = costs;
    const canvas = document.getElementById('cost-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width, H = rect.height;
    const entries = Object.entries(costs).filter(function(e) { return e[1] > 0; });
    if (entries.length === 0) { drawChart(); return; }

    const pad = { top: 16, right: 10, bottom: 32, left: 10 };
    const maxCost = Math.max.apply(null, entries.map(function(e) { return e[1]; }));
    const slotW = (W - pad.left - pad.right) / entries.length;
    const barW = Math.max(8, slotW - 4);

    ctx.clearRect(0, 0, W, H);

    entries.forEach(function(entry, i) {
      const name = entry[0], val = entry[1];
      const x = pad.left + i * slotW;
      const barH = Math.max(2, (val / maxCost) * (H - pad.top - pad.bottom));
      const y = H - pad.bottom - barH;
      const isLocal = name.toLowerCase().includes('local') || name.toLowerCase().includes('ollama');
      ctx.fillStyle = isLocal ? '#ff87ff' : '#00ffff';
      ctx.fillRect(x, y, barW, barH);
      ctx.fillStyle = '#666';
      ctx.font = '9px JetBrains Mono';
      ctx.textAlign = 'center';
      const label = name.length > 8 ? name.slice(0, 7) + '…' : name;
      ctx.fillText(label, x + barW / 2, H - pad.bottom + 12);
    });
  }

  // ── Fetch Mode ──
  async function fetchHealth() {
    try {
      const r = await fetch('/health');
      const data = await r.json();
      document.getElementById('mode-badge').textContent = data.mode + '-mode';
    } catch(e) {}
  }

  // ── Cost Chart (Canvas) ──
  function drawChart() {
    const canvas = document.getElementById('cost-chart');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;
    const pad = { top: 20, right: 10, bottom: 20, left: 50 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);

    if (chartLocal.length < 2) return;

    const maxVal = Math.max(
      Math.max.apply(null, chartLocal),
      Math.max.apply(null, chartCloud),
      0.001
    );

    function xPos(i) { return pad.left + (i / (chartLocal.length - 1)) * plotW; }
    function yPos(v) { return pad.top + plotH - (v / maxVal) * plotH; }

    // Grid lines
    ctx.strokeStyle = '#2a2620';
    ctx.lineWidth = 0.5;
    for (var g = 0; g <= 4; g++) {
      var gy = pad.top + (g / 4) * plotH;
      ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(W - pad.right, gy); ctx.stroke();
      ctx.fillStyle = '#58534a';
      ctx.font = '10px JetBrains Mono';
      ctx.textAlign = 'right';
      ctx.fillText('$' + (maxVal * (1 - g / 4)).toFixed(4), pad.left - 6, gy + 3);
    }

    // Local line (accent)
    ctx.strokeStyle = '#ff87ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (var i = 0; i < chartLocal.length; i++) {
      var x = xPos(i), y = yPos(chartLocal[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Cloud line (cool)
    ctx.strokeStyle = '#00ffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (var j = 0; j < chartCloud.length; j++) {
      var cx = xPos(j), cy = yPos(chartCloud[j]);
      if (j === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
    }
    ctx.stroke();
  }

  // ── Poll Loop ──
  function poll() {
    fetchVram();
    fetchStats();
    fetchFeed();
    fetchProviders();
  }

  fetchHealth();
  poll();
  setInterval(poll, POLL_MS);
  setInterval(fetchHealth, 5000);
})();
</script>
</body>
</html>
'''