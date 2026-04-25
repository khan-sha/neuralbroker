# Firebender Integration

**Type:** Phase 1
**Setup:** `neuralbrok integrations setup firebender`

## What is Firebender?
AI coding agent / IDE integration for Firebender.

## How NeuralBroker Integrates
- Config file written: `.firebender/config.json`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup firebender
```

### Files Generated
`.firebender/config.json` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
