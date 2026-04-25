# CodeBuddy Integration

**Type:** Phase 2
**Setup:** `neuralbrok integrations setup codebuddy`

## What is CodeBuddy?
AI coding agent / IDE integration for CodeBuddy.

## How NeuralBroker Integrates
- Config file written: `.codebuddy/skills/neuralbroker.md`
- VRAM > 80%: routes to local Ollama
- VRAM ≤ 80%: falls back to configured cloud provider

## Setup
```bash
neuralbrok integrations setup codebuddy
```

### Files Generated
`.codebuddy/skills/neuralbroker.md` — Main configuration file.

### Example Workflow
Run `neuralbrok start` then open your IDE and start coding. NeuralBroker will handle the routing.
