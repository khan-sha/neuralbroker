# Kiro CLI Integration

**Type:** Phase 2
**Setup:** `neuralbrok integrations setup kiro-cli`

## What is Kiro CLI?
AI coding agent / IDE integration for Kiro CLI.

## How NeuralBroker Integrates
- Config file written: `.kiro/skills/neuralbroker.md`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup kiro-cli
```

### Files Generated
`.kiro/skills/neuralbroker.md` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
