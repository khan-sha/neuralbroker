# neuralbroker

<img src="logo.svg" width="280" alt="neuralbroker logo" />

### VRAM-aware LLM routing daemon · local-first · OpenAI-compatible

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![PyPI](https://img.shields.io/pypi/v/neuralbrok)](https://pypi.org/project/neuralbrok/)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![FastAPI](https://img.shields.io/badge/framework-FastAPI-009688)
[![Website](https://img.shields.io/badge/website-neuralbroker.space-amber)](https://neuralbroker.space)

NeuralBroker is an OpenAI-compatible routing daemon that sends LLM requests to your local runtimes first, then spills to cloud providers only when VRAM pressure or policy requires it. It keeps your existing SDK flow intact while reducing avoidable cloud spend by turning local hardware into a first-class inference backend.

## Quickstart

---

### Option A — pip install (recommended)

Works on macOS, Linux, and Windows.

```bash
pip install neuralbrok
neuralbrok setup
neuralbrok start
```

---

### Option B — Install Normally (from source)

**macOS / Linux:**
```bash
git clone https://github.com/khan-sha/neuralbroker.git
cd neuralbroker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
neuralbrok setup
neuralbrok start
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/khan-sha/neuralbroker.git
cd neuralbroker
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
neuralbrok setup
neuralbrok start
```

---

### Option C — Docker

```bash
git clone https://github.com/khan-sha/neuralbroker.git
cd neuralbroker
cp .env.example .env
docker compose up -d
```

Proxy: `http://localhost:8000/v1` · Dashboard: `http://localhost:8000/dashboard`

---

### Prerequisites by platform

| Platform | Required | Notes |
|---|---|---|
| macOS · Apple Silicon | Python 3.10+ · Ollama | Metal GPU · automatic unified memory detection |
| macOS · Intel | Python 3.10+ · Ollama | CPU inference · no VRAM pressure |
| Linux · NVIDIA | Python 3.10+ · Ollama · CUDA 11.8+ | Full VRAM telemetry via pynvml |
| Linux · AMD | Python 3.10+ · Ollama · ROCm 5.0+ | ROCm telemetry · llama.cpp recommended |
| Linux · CPU | Python 3.10+ · Ollama | CPU fallback · cloud spillover always active |
| Windows · NVIDIA | Python 3.10+ · Ollama · CUDA 11.8+ | WSL2 recommended for best performance |
| Windows · CPU | Python 3.10+ · Ollama | CPU fallback |
| Docker · any | Docker Desktop or Docker Engine | No Python install needed |

Ollama: [ollama.com/download](https://ollama.com/download)

One-line SDK change:

```python
client = OpenAI(base_url="http://localhost:8000/v1", api_key="nb_live_...")
```

## What setup does
When you run `neuralbrok setup`, the device detection module automatically profiles your hardware:
- Detects your GPU vendor, model, and available VRAM (or unified memory).
- Configures the optimal local runtime (Ollama for CUDA/Metal, llama.cpp for ROCm/CPU).
- Calculates a safe VRAM threshold to avoid out-of-memory errors.
- Estimates the local electricity cost (TDP) for accurate cloud-cost comparisons.
- Recommends the best quantized models that fit entirely within your memory.

## Supported hardware
- **NVIDIA**: RTX 30/40 series, or any CUDA 11.8+ compatible GPU.
- **Apple Silicon**: M1 through M4 (Base, Pro, Max, Ultra variants) via Metal unified memory.
- **AMD**: Radeon GPUs supporting ROCm 5.0+.
- **CPU-only**: Fallback mode for systems without a dedicated AI accelerator.

## How It Works

1. Point your OpenAI SDK `base_url` to NeuralBroker.
2. NeuralBroker polls local GPU state (VRAM/utilization) on a short interval.
3. Policy engine scores local and cloud providers per request.
4. Response streams back in OpenAI format with routing headers/metrics.

## Routing Modes

### cost-mode
Route local when VRAM is under threshold, otherwise spill to the cheapest cloud backend.

```yaml
routing:
  default_mode: cost
```

### speed-mode
Always route local for lowest path latency and strict local-only behavior.

```yaml
routing:
  default_mode: speed
```

### fallback-mode
Prefer local; fall back to cloud on OOM/error; resume local when healthy.

```yaml
routing:
  default_mode: fallback
```

## Providers

20 Pattern-A providers · 8 Pattern-B providers · 4 local runtimes · 32 total

*Model lists reflect April 2026. Provider catalogs change frequently — check each provider's docs for the latest available models.*

### Pattern A — OpenAI-compatible

| Provider | Notes |
|---|---|
| OpenAI | gpt-5.4, gpt-5.4-mini, gpt-5.4-nano — flagship for complex reasoning and coding |
| Groq | Llama 4 Maverick, Llama 4 Scout, Qwen3-32B, Llama 3.3 70B — fastest inference via LPU, best first spillover target |
| Together AI | DeepSeek V3.2, Llama 4 Maverick, Qwen3-235B, Gemini 3.1 Flash Lite — widest open model catalog |
| Cerebras | Llama 3.3 70B, Qwen3-32B, Qwen3-235B, GPT-OSS 120B — wafer-scale, 20x faster throughput than NVIDIA |
| DeepInfra | DeepSeek V3.2, Qwen3-235B, Llama 4, Mistral Small 4 — cheapest per-token on most open models |
| Fireworks AI | Llama 4 Maverick, DeepSeek V3, Qwen3 — fast inference with strong function calling on open models |
| Lepton AI | Llama 3.3 70B, Qwen3 variants — serverless GPU cloud |
| Novita AI | Qwen3 and DeepSeek variants at lowest market pricing |
| Hyperbolic | Llama 4, Qwen3-235B — decentralized GPU marketplace, competitive on 70B+ models |
| Mistral AI | Mistral Small 4, Mistral Large, Codestral — only source for first-party Mistral models |
| Kimi (Moonshot) | Kimi K2.6 — highest-ranked open weights model on Intelligence Index (score 54), 1M token context |
| DeepSeek | DeepSeek V3.2, DeepSeek V3.1, DeepSeek R1 — best price-to-performance for coding and reasoning |
| Qwen (DashScope) | Qwen3-235B-A22B, Qwen3-32B, Qwen3-Coder, QwQ-32B, Qwen3.5 — Alibaba's hybrid reasoning/instruct family |
| Yi (01.AI) | Yi-Lightning, Yi-Large — strong multilingual, competitive pricing |
| Baichuan | Baichuan4 — strongest Chinese language understanding |
| Zhipu (GLM-4) | GLM-5.1 (Reasoning), GLM-5 — top open-weights reasoning models, score 51 on Intelligence Index |
| Perplexity | Sonar Pro, Sonar — live web search built in, unique for online and RAG workloads |
| AI21 Labs | Jamba 1.5 Large, Jamba 1.5 Mini — SSM-Transformer hybrid, long context at low cost |
| OctoAI | Llama 4, Qwen3 variants — auto-scaling serverless, good for burst spillover |
| OpenRouter | DeepSeek V3, Llama 4 Maverick, Qwen3-235B and 100+ more — last-resort fallback only |

### Pattern B — Custom translation layer

| Provider | Notes |
|---|---|
| Anthropic | Claude Opus 4.7, Claude Sonnet 4.6, Claude Haiku 4.5 — best for agentic coding, long context, and complex reasoning |
| Google Gemini | Gemini 3.1 Pro, Gemini 3.1 Flash, Gemini 3 Flash Preview — top reasoning benchmarks, 1M token context |
| Cohere | Command-R+ — enterprise RAG with built-in grounding and citation |
| Replicate | Any open model including fine-tunes — polling-based predictions API, widest model selection |
| Cloudflare AI | Workers AI — edge inference at Cloudflare's global network, lowest geographic latency |
| AWS Bedrock | Claude Opus 4.7, Haiku 4.5, Llama 4, Amazon Nova 2 Pro/Lite/Micro — managed AWS, data residency |
| Azure OpenAI | gpt-5.4, gpt-5.4-mini — deployment-based, api-key auth, Microsoft enterprise agreements |
| Google Vertex | Gemini 3.1 Pro via GCP — VPC and private endpoint support for Google Cloud teams |

### Local runtimes

| Runtime | Platform | Notes |
|---|---|---|
| Ollama | NVIDIA · Apple Silicon · AMD | Recommended default — native Metal and CUDA, Llama 4, Qwen3, DeepSeek models in model library |
| llama.cpp | NVIDIA · AMD · CPU | Best for AMD ROCm, CPU-only, and maximum quantization control — supports Qwen3, Llama 4, DeepSeek |
| LM Studio | NVIDIA · Apple Silicon | GUI-first model browser — exposes OpenAI-compatible server, good for Apple Silicon |
| vLLM | NVIDIA | Best throughput for concurrent requests — PagedAttention, continuous batching, production serving |

## Docker

```bash
docker compose up -d
```

This starts NeuralBroker plus supporting observability services (Prometheus/Grafana) defined in `docker-compose.yml`.

## Development

Run locally:

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Run tests:

```bash
pytest -q
```

## Contributing

Contributions are welcome. If you have bug reports, routing ideas, provider integrations, or docs improvements, open an issue or PR and we will review quickly: [GitHub Issues](https://github.com/khan-sha/neuralbroker/issues).

MIT License — see `LICENSE`.
