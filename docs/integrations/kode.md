# Kode Integration

**Type:** Phase 2
**Setup:** `neuralbrok integrations setup kode`

## What is Kode?
AI coding agent / IDE integration for Kode.

## How NeuralBroker Integrates
- Config file written: `.kode/skills/neuralbroker.md`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup kode
```

### Files Generated
`.kode/skills/neuralbroker.md` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
