# Cortex Code Integration

**Type:** Phase 2
**Setup:** `neuralbrok integrations setup cortex-code`

## What is Cortex Code?
AI coding agent / IDE integration for Cortex Code.

## How NeuralBroker Integrates
- Config file written: `.cortex/skills/neuralbroker.md`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup cortex-code
```

### Files Generated
`.cortex/skills/neuralbroker.md` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
