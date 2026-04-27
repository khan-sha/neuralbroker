# NeuralBroker

<img src="logo.svg" width="280" alt="NeuralBroker logo" />

### Route every AI request through subscriptions you already own. No API keys. No new bills.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![PyPI](https://img.shields.io/pypi/v/neuralbrok)](https://pypi.org/project/neuralbrok/)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![FastAPI](https://img.shields.io/badge/framework-FastAPI-009688)
[![Website](https://img.shields.io/badge/website-neuralbroker.space-pink)](https://neuralbroker.space)

---

## The idea in one sentence

You already pay for Claude Pro. NeuralBroker lets **every app on your machine** use it — automatically, for free, without ever touching an API key.

```
pip install neuralbrok
neuralbrok start
```

That's it. NeuralBroker auto-detects your installed Claude Code OAuth session, ChatGPT auth, Ollama models, and any environment API keys — then presents them as a single OpenAI-compatible endpoint at `localhost:8000/v1`. Point any tool at that URL and routing is live.

---

## What no one else does

### 1. Subscription inheritance — turn a $20/month sub into a free API

NeuralBroker reads the OAuth session your Claude Code CLI already holds in `~/.claude/.credentials.json`. It shells out to the `claude` binary to answer requests. No token copying. No API key. Your Claude Pro or Max subscription covers it at zero marginal cost.

```
Auto-discovered on startup:
  ✓ Claude PRO subscription    ~/.claude/.credentials.json
  ✓ Ollama (v0.20.4)           localhost:11434
  ✓ ChatGPT subscription       ~/.codex/auth.json  (roadmap)
```

Any app that speaks OpenAI format now routes through your subscription:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="any-string")
response = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Hello"}]
)
# Routed through your Claude Pro subscription. Cost: $0.
```

### 2. VRAM-aware local-first routing — your GPU before the cloud

NeuralBroker polls GPU state every 500ms. Requests go to your local Ollama models first. Cloud kicks in only when VRAM is under pressure or local fails. Your electricity bill, not Anthropic's servers.

```
Request arrives → check VRAM → qwen2.5:0.5b fits → route local → $0.00001
Request arrives → VRAM full → spill → claude_code subprocess → $0.00000 (Pro sub)
Request arrives → no local match → spill → cheapest cloud provider
```

### 3. Zero config — discovers everything automatically

No yaml required. On first start NeuralBroker:

- Reads `~/.claude/.credentials.json` → registers Claude Pro/Max via subprocess
- Reads `~/.codex/auth.json` → registers OpenAI or ChatGPT tokens
- Scans environment for `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, etc.
- Pings `localhost:11434` → registers Ollama if running
- Pings `localhost:8080` → registers llama.cpp if running

Then picks the cheapest path per request. You never write a config file.

### 4. Claude Code + Ollama hybrid — one endpoint for your whole dev stack

```bash
# NeuralBroker running in the background
neuralbrok start

# Claude Code routes through NB (simple tasks → local, hard tasks → Claude Pro)
ANTHROPIC_BASE_URL=http://localhost:8000/v1 claude

# Cursor, Cline, Codex — same endpoint
# neuralbrok integrations setup cursor
# neuralbrok integrations setup codex
```

Simple coding tasks → `qwen2.5:0.5b` locally, instant, free.
Complex reasoning → `claude-sonnet-4-6` via your Pro subscription, free.
Cloud overflow → cheapest available provider.

---

## Quickstart

### Option A — pip install

```bash
pip install neuralbrok
neuralbrok start
```

### Option B — from source

**macOS / Linux:**
```bash
git clone https://github.com/khan-sha/neuralbroker.git
cd neuralbroker
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
neuralbrok start
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/khan-sha/neuralbroker.git
cd neuralbroker
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -e .
neuralbrok start
```

### Option C — Docker

```bash
git clone https://github.com/khan-sha/neuralbroker.git
cd neuralbroker
cp .env.example .env
docker compose up -d
```

Proxy: `http://localhost:8000/v1` · Dashboard: `http://localhost:8000/dashboard`

---

## Prerequisites

| Platform | Required | Notes |
|---|---|---|
| macOS · Apple Silicon | Python 3.10+ · Ollama | Metal GPU · automatic unified memory detection |
| macOS · Intel | Python 3.10+ · Ollama | CPU inference · no VRAM pressure |
| Linux · NVIDIA | Python 3.10+ · Ollama · CUDA 11.8+ | Full VRAM telemetry via pynvml |
| Linux · AMD | Python 3.10+ · Ollama · ROCm 5.0+ | ROCm telemetry · llama.cpp recommended |
| Linux · CPU | Python 3.10+ · Ollama | CPU fallback · cloud spillover active |
| Windows · NVIDIA | Python 3.10+ · Ollama · CUDA 11.8+ | WSL2 recommended |
| Windows · CPU | Python 3.10+ · Ollama | CPU fallback |
| Docker · any | Docker Desktop or Engine | No Python install needed |

Ollama: [ollama.com/download](https://ollama.com/download)

---

## Subscription inheritance — detailed

NeuralBroker auto-discovers auth in priority order:

| Source | Provider | Type | Cost |
|---|---|---|---|
| `~/.claude/.credentials.json` | Claude Pro/Max via `claude` CLI | OAuth bearer | **$0** |
| `~/.codex/auth.json` (tokens) | ChatGPT via Codex | OAuth bearer | **$0** (roadmap) |
| `~/.codex/auth.json` (api_key) | OpenAI | API key | per-token |
| `ANTHROPIC_API_KEY` | Anthropic API | API key | per-token |
| `OPENAI_API_KEY` | OpenAI API | API key | per-token |
| `GROQ_API_KEY` | Groq | API key | per-token |
| `localhost:11434` | Ollama | none | **electricity only** |
| `localhost:8080` | llama.cpp | none | **electricity only** |

Subscription tokens always override env API keys for the same provider. They're effectively free at the margin — the subscription is already paid.

Check what was discovered:

```bash
curl http://localhost:8000/nb/discovered
```

Disable auto-discovery:

```bash
NB_DISABLE_AUTO_DISCOVERY=1 neuralbrok start
```

---

## Routing modes

### cost (default)
Routes local when VRAM free. Spills to cheapest cloud (subscription first, then cheapest paid).

### speed
Always local for minimum latency. Cloud only on local failure.

### fallback
Prefer local. Fall back on OOM or repeated error within 30s. Resumes local when healthy.

### smart
Classifies each prompt (code, reasoning, RAG, fast response, long context, tools, chat). Runs SmartModelSelector to score all runnable local models. Picks best match. Falls back to cloud only on failure.

```yaml
routing:
  default_mode: cost  # cost | speed | fallback | smart
```

---

## SmartModelSelector

Scores every model that fits in available VRAM per request:

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

Scores normalized 0–100%. Top scorer routes.

---

## API

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat completions |
| `POST /v1/messages` | Anthropic-compatible messages (auto-translated) |
| `POST /v1/completions` | Text completions |
| `GET /health` | Status + uptime |
| `GET /nb/stats` | Routing stats, fallbacks, avg latency |
| `GET /nb/discovered` | Auto-discovered auth state |
| `GET /metrics` | Prometheus scrape endpoint |
| `GET /dashboard` | Live routing dashboard |

---

## Providers

32 total: 20 Pattern-A (OpenAI-compatible) · 8 Pattern-B (custom adapter) · 4 local runtimes · Claude Code subprocess

### Subscription-backed (no API key)

| Provider | Auth source | Cost |
|---|---|---|
| Claude Pro/Max via `claude` CLI | `~/.claude/.credentials.json` | $0 |
| ChatGPT via Codex (roadmap) | `~/.codex/auth.json` | $0 |
| Ollama (all local models) | localhost:11434 | electricity |
| llama.cpp | localhost:8080 | electricity |

### Pattern A — OpenAI-compatible

| Provider | Notes |
|---|---|
| OpenAI | GPT-4o, GPT-4o mini |
| Groq | Llama 4, Qwen3-32B — fastest via LPU |
| Together AI | DeepSeek V3, Llama 4, Qwen3-235B |
| Cerebras | Llama 3.3 70B — wafer-scale throughput |
| DeepInfra | DeepSeek V3, Qwen3-235B — cheapest per-token |
| Fireworks AI | Llama 4, DeepSeek V3, Qwen3 |
| Lepton AI | Llama 3.3 70B, Qwen3 — serverless GPU |
| Novita AI | Qwen3 and DeepSeek at lowest pricing |
| Hyperbolic | Llama 4, Qwen3-235B — decentralized GPU |
| Mistral AI | Mistral Small, Large, Codestral |
| Kimi (Moonshot) | Kimi K2 — 1M token context |
| DeepSeek | DeepSeek V3, R1 — best price/performance for coding |
| Qwen (DashScope) | Qwen3-235B-A22B, Qwen3-Coder, QwQ-32B |
| Yi (01.AI) | Yi-Lightning — strong multilingual |
| Baichuan | Baichuan4 — Chinese language |
| Zhipu (GLM-4) | GLM-4 |
| Perplexity | Sonar Pro — live web search built in |
| AI21 Labs | Jamba 1.5 — SSM-Transformer, long context |
| OctoAI | Auto-scaling serverless burst |
| OpenRouter | 100+ models — last-resort fallback |

### Pattern B — Custom translation layer

| Provider | Notes |
|---|---|
| Anthropic (API key) | Claude Sonnet, Haiku — when no Pro sub |
| Google Gemini | Gemini 1.5 Pro, Flash — 1M token context |
| Cohere | Command-R+ — enterprise RAG |
| Replicate | Any open model including fine-tunes |
| Cloudflare AI | Edge inference, global |
| AWS Bedrock | Claude, Llama 4, Amazon Nova — data residency |
| Azure OpenAI | GPT-4o — Microsoft enterprise |
| Google Vertex | Gemini via GCP — private endpoint |

---

## Integrations

23 AI coding agents auto-configured via `neuralbrok integrations setup <name>`:

| Agent | Config | Command |
|---|---|---|
| Claude Code | `.claude/settings.json` | `neuralbrok integrations setup claude-code` |
| Cursor | `.cursor/mcp.json` | `neuralbrok integrations setup cursor` |
| Cline | `.cline/settings.json` | `neuralbrok integrations setup cline` |
| GitHub Copilot | `.vscode/settings.json` | `neuralbrok integrations setup github-copilot` |
| Gemini CLI | `.gemini/settings.json` | `neuralbrok integrations setup gemini-cli` |
| OpenCode | `opencode.json` | `neuralbrok integrations setup opencode` |
| Warp | `~/.warp/preferences.yaml` | `neuralbrok integrations setup warp` |
| Codex | `.env + ~/.codex/config.json` | `neuralbrok integrations setup codex` |
| Amp | `~/.amp/config.json` | `neuralbrok integrations setup amp` |
| Kimi Code | `.kimi/config.json` | `neuralbrok integrations setup kimi-code` |
| Firebender | `.firebender/config.json` | `neuralbrok integrations setup firebender` |
| Deep Agents | `.deepagent/config.json` | `neuralbrok integrations setup deep-agents` |
| Windsurf, Trae, Cursor, Kilo Code, Qwen Code + more | skill files | `neuralbrok integrations setup <name>` |

---

## Setup TUI (optional)

`neuralbrok setup` runs a guided terminal UI for users who want manual configuration:

- Detects GPU vendor, model, VRAM (or Apple unified memory)
- Profiles every installed Ollama model — VRAM fit, tok/s, compatibility score
- Manual mode: select exact models by number from ranked list
- Algorithm self-test: shows which model routes to 4 sample prompts and why
- Writes `~/.neuralbrok/config.yaml`

---

## Roadmap

### Shipped

- [x] Zero-config subscription inheritance — Claude Pro/Max OAuth auto-discovery, no API key
- [x] Claude Code subprocess provider — Pro subscription as a free inference backend
- [x] `/v1/messages` Anthropic endpoint — full wire format translation
- [x] `/nb/discovered` — inspect auto-discovered auth at runtime
- [x] VRAM-aware routing — GPU polling, 4 routing modes, cost formula
- [x] SmartModelSelector — per-request model scoring
- [x] Dashboard v2 — live routing waterfall, mode switching, cost graph
- [x] `neuralbrok doctor` — diagnose config, test connectivity, benchmark local
- [x] `neuralbrok code` — launch Claude Code with NB routing context
- [x] 23 agent integrations auto-configured via CLI

### Phase 2

- [ ] Subagent decomposition — planner model breaks task into subtasks routed to different models, synthesizer merges
- [ ] Model council — N models debate a response, moderator merges, disagreement logged as metadata
- [ ] ChatGPT subscription routing — chatgpt.com OAuth via Codex for $0 GPT-4o
- [ ] Prompt caching — detect repeated system prompts, route to providers with cache discount
- [ ] Per-model cost tracking — log token spend per model per day, budget alerts

### Phase 3

- [ ] Dynamic provider weighting — auto-demote slow or error-prone providers
- [ ] Fine-grained routing rules — route by model name, tag, or regex in config.yaml
- [ ] Multi-GPU VRAM aggregation — per-GPU model pinning for multi-card setups
- [ ] Privacy-tier routing — PII detection forces local-only path
- [ ] Token budget enforcement — daily $ cap auto-downgrades models

---

## Development

```bash
uvicorn src.neuralbrok.main:app --reload --host 0.0.0.0 --port 8000
pytest -q
```

---

## Contributing

Bug reports, routing ideas, provider integrations, docs: [GitHub Issues](https://github.com/khan-sha/neuralbroker/issues)

MIT License — see `LICENSE`.
