# Kilo Code Integration

**Type:** Phase 2
**Setup:** `neuralbrok integrations setup kilo-code`

## What is Kilo Code?
AI coding agent / IDE integration for Kilo Code.

## How NeuralBroker Integrates
- Config file written: `.kilocode/skills/neuralbroker.md`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup kilo-code
```

### Files Generated
`.kilocode/skills/neuralbroker.md` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
