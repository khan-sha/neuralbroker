# Amp Integration

**Type:** Phase 1
**Setup:** `neuralbrok integrations setup amp`

## What is Amp?
AI coding agent / IDE integration for Amp.

## How NeuralBroker Integrates
- Config file written: `~/.amp/config.json`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup amp
```

### Files Generated
`~/.amp/config.json` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
