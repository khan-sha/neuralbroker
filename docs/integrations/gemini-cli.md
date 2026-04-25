# Gemini CLI Integration

**Type:** Phase 1
**Setup:** `neuralbrok integrations setup gemini-cli`

## What is Gemini CLI?
AI coding agent / IDE integration for Gemini CLI.

## How NeuralBroker Integrates
- Config file written: `.gemini/settings.json`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup gemini-cli
```

### Files Generated
`.gemini/settings.json` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
