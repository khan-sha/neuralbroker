# Augment Integration

**Type:** Phase 2
**Setup:** `neuralbrok integrations setup augment`

## What is Augment?
AI coding agent / IDE integration for Augment.

## How NeuralBroker Integrates
- Config file written: `.augment/skills/neuralbroker.md`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup augment
```

### Files Generated
`.augment/skills/neuralbroker.md` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
