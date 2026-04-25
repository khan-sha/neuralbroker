# Cursor Integration

**Type:** Phase 1
**Setup:** `neuralbrok integrations setup cursor`

## What is Cursor?
AI coding agent / IDE integration for Cursor.

## How NeuralBroker Integrates
- Config file written: `.cursor/mcp.json`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup cursor
```

### Files Generated
`.cursor/mcp.json` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
