# GitHub Copilot Integration

**Type:** Phase 1
**Setup:** `neuralbrok integrations setup github-copilot`

## What is GitHub Copilot?
AI coding agent / IDE integration for GitHub Copilot.

## How NeuralBroker Integrates
- Config file written: `.vscode/settings.json`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup github-copilot
```

### Files Generated
`.vscode/settings.json` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
