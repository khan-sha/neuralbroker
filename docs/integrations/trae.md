# Trae Integration

**Type:** Phase 2
**Setup:** `neuralbrok integrations setup trae`

## What is Trae?
AI coding agent / IDE integration for Trae.

## How NeuralBroker Integrates
- Config file written: `.trae/skills/neuralbroker.md`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup trae
```

### Files Generated
`.trae/skills/neuralbroker.md` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
