# Claude Code Integration

**Type:** Phase 1
**Setup:** `neuralbrok integrations setup claude-code`

## What is Claude Code?
AI coding agent / IDE integration for Claude Code.

## How NeuralBroker Integrates
- Config file written: `.claude/settings.json`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup claude-code
```

### Files Generated
`.claude/settings.json` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
