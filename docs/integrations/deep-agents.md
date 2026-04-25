# Deep Agents Integration

**Type:** Phase 1
**Setup:** `neuralbrok integrations setup deep-agents`

## What is Deep Agents?
AI coding agent / IDE integration for Deep Agents.

## How NeuralBroker Integrates
- Config file written: `.deepagent/config.json`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup deep-agents
```

### Files Generated
`.deepagent/config.json` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
