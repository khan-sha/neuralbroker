# Kimi Code Integration

**Type:** Phase 1
**Setup:** `neuralbrok integrations setup kimi-code`

## What is Kimi Code?
AI coding agent / IDE integration for Kimi Code.

## How NeuralBroker Integrates
- Config file written: `.kimi/config.json`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup kimi-code
```

### Files Generated
`.kimi/config.json` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
