# neuralbroker

<img src="logo.svg" width="280" alt="neuralbroker logo" />

### VRAM-aware LLM routing daemon · local-first · OpenAI-compatible

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![PyPI](https://img.shields.io/pypi/v/neuralbrok)](https://pypi.org/project/neuralbrok/)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![FastAPI](https://img.shields.io/badge/framework-FastAPI-009688)
[![Website](https://img.shields.io/badge/website-neuralbroker.space-pink)](https://neuralbroker.space)

NeuralBroker is an OpenAI-compatible routing daemon that sends LLM requests to your local runtimes first, then spills to cloud providers only when VRAM pressure or policy requires it. It keeps your existing SDK flow intact while reducing avoidable cloud spend by turning local hardware into a first-class inference backend.

---

## What makes it different

- **VRAM-aware routing** — polls GPU state every 500ms; routes locally when memory is free, spills to cloud when pressure builds
- **SmartModelSelector** — scores every runnable local model on params, workload fit, tok/s, context length, VRAM headroom, and MoE architecture; picks the best one per request category, not just the largest
- **MoE detection** — identifies mixture-of-experts models (e.g. `qwen3:30b-a3b`) and scores them separately from dense models; prioritizes them for fast-response workloads where low activation count wins
- **Four routing modes** — cost, speed, fallback, and smart (prompt-classified, model-matched)
- **Manual model selection** — technical users can pick exact local models by number during setup; no guessing
- **32 providers, zero SDK changes** — point `base_url` at NeuralBroker, keep the same `openai` client
- **Interactive setup TUI** — hardware detection, VRAM visualization, model compatibility bars, algorithm self-test, and routing mode selection all in one guided flow

---

## Quickstart

### Option A — pip install (recommended)

Works on macOS, Linux, and Windows.

```bash
pip install neuralbrok
neuralbrok setup
neuralbrok start
```

---

### Option B — from source

**macOS / Linux:**
```bash
git clone https://github.com/khan-sha/neuralbroker.git
cd neuralbroker
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
neuralbrok setup
neuralbrok start
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/khan-sha/neuralbroker.git
cd neuralbroker
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
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

---

## What setup does

`neuralbrok setup` runs a fully interactive TUI that:

- Detects GPU vendor, model, and available VRAM (or Apple unified memory)
- Shows a color-coded VRAM bar and backend (CUDA / Metal / ROCm / CPU)
- Profiles every installed Ollama model — VRAM fit, estimated tok/s, compatibility score
- Lets you pick workload type (code, chat, reasoning, RAG, mixed, etc.) or use **manual mode** to select exact models by number from a ranked list
- Runs an algorithm self-test showing which model routes to 4 sample prompts and why
- Prompts for cloud provider API keys (optional — local-only works without any)
- Writes `~/.neuralbrok/config.yaml`

---

## Supported hardware

- **NVIDIA**: RTX 30/40 series, or any CUDA 11.8+ compatible GPU
- **Apple Silicon**: M1 through M4 (Base, Pro, Max, Ultra variants) via Metal unified memory
- **AMD**: Radeon GPUs supporting ROCm 5.0+
- **CPU-only**: Fallback mode for systems without a dedicated AI accelerator

---

## How it works

1. Point your OpenAI SDK `base_url` to NeuralBroker.
2. NeuralBroker polls local GPU state (VRAM / utilization) every 500ms.
3. Policy engine scores local and cloud providers per request using the active routing mode.
4. Response streams back in OpenAI format with routing headers and metrics.

---

## Routing modes

### cost
Route local when VRAM is under threshold, otherwise spill to the cheapest cloud backend.

```yaml
routing:
  default_mode: cost
```

### speed
Always route local for lowest path latency. Cloud only on local failure.

```yaml
routing:
  default_mode: speed
```

### fallback
Prefer local. Fall back to cloud on OOM or repeated error within a 30s window. Resume local when healthy.

```yaml
routing:
  default_mode: fallback
```

### smart
Classify each prompt into a workload category (code, reasoning, RAG, fast response, long context, tools, chat). Run SmartModelSelector to score all runnable local models against that category using params, workload fit, tok/s, context length, VRAM headroom, and MoE architecture weight. Pick the best match. Fall back to cloud only on failure.

```yaml
routing:
  default_mode: smart
```

---

## SmartModelSelector

The selector scores every model that fits in available VRAM and ranks them per request:

| Signal | Weight |
|---|---|
| Parameter count | baseline score |
| Workload capability match | +15 per matching tag |
| Workload recommended_for match | +20 per matching tag |
| tok/s on your hardware (>60) | +10 |
| tok/s on your hardware (>30) | +5 |
| tok/s on your hardware (<10) | −10 |
| Long-context request + ctx ≥128k | +25 |
| VRAM headroom (free − model) | ×2 |
| MoE model + fast_response workload | +15 |

Scores are normalized to 0–100% and the top 3 are returned. The highest scorer routes.

---

## API endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat completions |
| `POST /v1/completions` | OpenAI-compatible text completions |
| `GET /health` | Health check — returns status and uptime |
| `GET /nb/stats` | Routing stats — requests routed, fallbacks, smart classifications, avg classify ms |
| `GET /metrics` | Prometheus metrics scrape endpoint |
| `GET /dashboard` | Local web dashboard — live routing view |

---

## Providers

32 total: 20 Pattern-A (OpenAI-compatible) · 8 Pattern-B (custom adapter) · 4 local runtimes

*Model lists reflect current catalog at time of release. Provider catalogs change frequently — check each provider's docs for latest available models.*

### Pattern A — OpenAI-compatible

| Provider | Notes |
|---|---|
| OpenAI | GPT-4o, GPT-4o mini — flagship for complex reasoning and coding |
| Groq | Llama 4 Maverick, Llama 4 Scout, Qwen3-32B, Llama 3.3 70B — fastest inference via LPU, best first spillover target |
| Together AI | DeepSeek V3, Llama 4 Maverick, Qwen3-235B — widest open model catalog |
| Cerebras | Llama 3.3 70B, Qwen3-32B — wafer-scale, 20x faster throughput than NVIDIA |
| DeepInfra | DeepSeek V3, Qwen3-235B, Llama 4, Mistral Small — cheapest per-token on most open models |
| Fireworks AI | Llama 4 Maverick, DeepSeek V3, Qwen3 — fast inference with strong function calling on open models |
| Lepton AI | Llama 3.3 70B, Qwen3 variants — serverless GPU cloud |
| Novita AI | Qwen3 and DeepSeek variants at lowest market pricing |
| Hyperbolic | Llama 4, Qwen3-235B — decentralized GPU marketplace, competitive on 70B+ models |
| Mistral AI | Mistral Small, Mistral Large, Codestral — only source for first-party Mistral models |
| Kimi (Moonshot) | Kimi K2 — strong long-context (1M token) and multilingual |
| DeepSeek | DeepSeek V3, DeepSeek R1 — best price-to-performance for coding and reasoning |
| Qwen (DashScope) | Qwen3-235B-A22B, Qwen3-32B, Qwen3-Coder, QwQ-32B — Alibaba's hybrid reasoning/instruct family |
| Yi (01.AI) | Yi-Lightning, Yi-Large — strong multilingual, competitive pricing |
| Baichuan | Baichuan4 — strongest Chinese language understanding |
| Zhipu (GLM-4) | GLM-4 — strong open-weights reasoning |
| Perplexity | Sonar Pro, Sonar — live web search built in, unique for online and RAG workloads |
| AI21 Labs | Jamba 1.5 Large, Jamba 1.5 Mini — SSM-Transformer hybrid, long context at low cost |
| OctoAI | Llama 4, Qwen3 variants — auto-scaling serverless, good for burst spillover |
| OpenRouter | 100+ models — last-resort fallback with widest model selection |

### Pattern B — Custom translation layer

| Provider | Notes |
|---|---|
| Anthropic | Claude Sonnet, Claude Haiku — best for agentic coding, long context, and complex reasoning |
| Google Gemini | Gemini 1.5 Pro, Gemini 1.5 Flash — top reasoning benchmarks, 1M token context |
| Cohere | Command-R+ — enterprise RAG with built-in grounding and citation |
| Replicate | Any open model including fine-tunes — polling-based predictions API, widest model selection |
| Cloudflare AI | Workers AI — edge inference at Cloudflare's global network, lowest geographic latency |
| AWS Bedrock | Claude, Llama 4, Amazon Nova — managed AWS, data residency |
| Azure OpenAI | GPT-4o, GPT-4o mini — deployment-based, Microsoft enterprise agreements |
| Google Vertex | Gemini Pro via GCP — VPC and private endpoint support for Google Cloud teams |

### Local runtimes

| Runtime | Platform | Notes |
|---|---|---|
| Ollama | NVIDIA · Apple Silicon · AMD | Recommended default — native Metal and CUDA, Llama 4, Qwen3, DeepSeek in model library |
| llama.cpp | NVIDIA · AMD · CPU | Best for AMD ROCm, CPU-only, and maximum quantization control |
| LM Studio | NVIDIA · Apple Silicon | GUI-first model browser — exposes OpenAI-compatible server |
| vLLM | NVIDIA | Best throughput for concurrent requests — PagedAttention, continuous batching, production serving |

---

## Docker

```bash
docker compose up -d
```

Starts NeuralBroker plus Prometheus and Grafana for observability. Configure credentials in `.env` (see `.env.example`).

---

## Development

```bash
uvicorn src.neuralbrok.main:app --reload --host 0.0.0.0 --port 8000
```

Run tests:

```bash
pytest -q
```

---

## Roadmap

### Phase 1 (BETA)
- [ ] **Claude Code terminal connection** — `neuralbrok code` runs NeuralBroker-aware Claude Code shell with routing context
- [ ] Dashboard v2 — live routing waterfall, model switching, per-provider cost graph
- [ ] `neuralbrok doctor` — diagnose config issues, test provider connectivity, benchmark local models

### Phase 2
- [ ] **Hermes agent integration** — deploy autonomous agents that use NeuralBroker for model selection and routing
- [ ] **Openclaw integration** — connect Openclaw orchestrator to NeuralBroker for decentralized agent coordination
- [ ] Prompt caching integration — detect repeated system prompts and route to providers with cache discount
- [ ] Per-model cost tracking — log actual token spend per model per day with budget alerts

### Phase 3
- [ ] Dynamic provider weighting — auto-demote slow or error-prone providers without manual config changes
- [ ] Fine-grained routing rules — route by model name, tag, or regex in config.yaml
- [ ] GGUF download helper — pull and register quantized models from HuggingFace directly from CLI
- [ ] Multi-GPU support — VRAM aggregation and per-GPU model pinning for multi-card setups
- [ ] **Multi-agent framework support** — extend to Anthropic Managed Agents, LangGraph, CrewAI

---

## Contributing

Contributions are welcome. Bug reports, routing ideas, provider integrations, docs improvements — open an issue or PR: [GitHub Issues](https://github.com/khan-sha/neuralbroker/issues).

MIT License — see `LICENSE`.
