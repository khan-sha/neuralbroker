# OpenCode Integration

**Type:** Phase 1
**Setup:** `neuralbrok integrations setup opencode`

## What is OpenCode?
AI coding agent / IDE integration for OpenCode.

## How NeuralBroker Integrates
- Config file written: `opencode.json`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup opencode
```

### Files Generated
`opencode.json` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
