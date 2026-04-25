# Cline Integration

**Type:** Phase 1
**Setup:** `neuralbrok integrations setup cline`

## What is Cline?
AI coding agent / IDE integration for Cline.

## How NeuralBroker Integrates
- Config file written: `.cline/settings.json`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup cline
```

### Files Generated
`.cline/settings.json` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
