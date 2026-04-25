# Codex Integration

**Type:** Phase 1
**Setup:** `neuralbrok integrations setup codex`

## What is Codex?
AI coding agent / IDE integration for Codex.

## How NeuralBroker Integrates
- Config file written: `~/.codex/config.json`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup codex
```

### Files Generated
`~/.codex/config.json` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
