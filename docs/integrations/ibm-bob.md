# IBM Bob Integration

**Type:** Phase 2
**Setup:** `neuralbrok integrations setup ibm-bob`

## What is IBM Bob?
AI coding agent / IDE integration for IBM Bob.

## How NeuralBroker Integrates
- Config file written: `.bob/skills/neuralbroker.md`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup ibm-bob
```

### Files Generated
`.bob/skills/neuralbroker.md` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
