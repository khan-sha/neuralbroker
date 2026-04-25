# Warp Integration

**Type:** Phase 1
**Setup:** `neuralbrok integrations setup warp`

## What is Warp?
AI coding agent / IDE integration for Warp.

## How NeuralBroker Integrates
- Config file written: `~/.warp/preferences.yaml`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup warp
```

### Files Generated
`~/.warp/preferences.yaml` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
