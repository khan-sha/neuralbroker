"""
Custom YAML-based agent loader.

Loads agent definitions from ~/.neuralbrok/agents/*.yaml,
allowing users to define their own specialized agents.
"""
import logging
from pathlib import Path
from typing import Optional

import yaml

from neuralbrok.agents.builtin import AgentDef

logger = logging.getLogger(__name__)

AGENTS_DIR = Path.home() / ".neuralbrok" / "agents"


def load_custom_agents(agents_dir: Optional[Path] = None) -> list[AgentDef]:
    """Load custom agent definitions from YAML files.

    Each YAML file defines one agent with the following schema:

    ```yaml
    slug: my-agent
    name: My Custom Agent
    role: Does custom things
    system_prompt: |
      You are a custom agent that...
    capabilities:
      - coding
      - analysis
    preferred_model_tags:
      - qwen3
    preferred_use_case: coding
    icon: "🔧"
    color: "#ff87ff"
    temperature: 0.5
    max_tokens: 8192
    tools_enabled: true
    ```
    """
    if agents_dir is None:
        agents_dir = AGENTS_DIR

    if not agents_dir.exists():
        return []

    agents: list[AgentDef] = []

    for yaml_file in sorted(agents_dir.glob("*.yaml")):
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data or not isinstance(data, dict):
                continue

            # Validate required fields
            if not data.get("slug") or not data.get("system_prompt"):
                logger.warning(f"Skipping {yaml_file.name}: missing slug or system_prompt")
                continue

            agent = AgentDef(
                slug=data["slug"],
                name=data.get("name", data["slug"].replace("-", " ").title()),
                role=data.get("role", "Custom agent"),
                system_prompt=data["system_prompt"],
                capabilities=data.get("capabilities", ["chat"]),
                preferred_model_tags=data.get("preferred_model_tags", []),
                preferred_use_case=data.get("preferred_use_case", "general"),
                icon=data.get("icon", "🔧"),
                color=data.get("color", "#ff87ff"),
                max_tokens=data.get("max_tokens", 8192),
                temperature=data.get("temperature", 0.7),
                tools_enabled=data.get("tools_enabled", True),
            )
            agents.append(agent)
            logger.info(f"Loaded custom agent: {agent.slug} from {yaml_file.name}")

        except Exception as e:
            logger.warning(f"Failed to load agent from {yaml_file.name}: {e}")

    return agents


def save_agent(agent: AgentDef, agents_dir: Optional[Path] = None) -> Path:
    """Save an agent definition to a YAML file."""
    if agents_dir is None:
        agents_dir = AGENTS_DIR

    agents_dir.mkdir(parents=True, exist_ok=True)
    path = agents_dir / f"{agent.slug}.yaml"

    data = {
        "slug": agent.slug,
        "name": agent.name,
        "role": agent.role,
        "system_prompt": agent.system_prompt,
        "capabilities": agent.capabilities,
        "preferred_model_tags": agent.preferred_model_tags,
        "preferred_use_case": agent.preferred_use_case,
        "icon": agent.icon,
        "color": agent.color,
        "max_tokens": agent.max_tokens,
        "temperature": agent.temperature,
        "tools_enabled": agent.tools_enabled,
    }

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return path
