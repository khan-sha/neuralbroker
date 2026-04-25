# OpenClaw Integration

**Type:** Phase 2
**Setup:** `neuralbrok integrations setup openclaw`

## What is OpenClaw?
AI coding agent / IDE integration for OpenClaw.

## How NeuralBroker Integrates
- Config file written: `skills/neuralbroker.md`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup openclaw
```

### Files Generated
`skills/neuralbroker.md` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
