"""
Agent system for NeuralBroker v2.0

Provides the agent registry, built-in agent definitions,
and YAML-based custom agent loading.
"""
from neuralbrok.agents.builtin import BUILTIN_AGENTS, AgentDef
from neuralbrok.agents.custom import load_custom_agents

# Merged registry: built-in + custom (custom can override built-in)
_registry: dict[str, AgentDef] | None = None


def get_agent_registry() -> dict[str, AgentDef]:
    """Get the full agent registry (built-in + custom), cached after first load."""
    global _registry
    if _registry is None:
        _registry = {a.slug: a for a in BUILTIN_AGENTS}
        try:
            customs = load_custom_agents()
            _registry.update({a.slug: a for a in customs})
        except Exception:
            pass  # Custom agents are optional
    return _registry


def get_agent(slug: str) -> AgentDef | None:
    """Look up an agent by slug."""
    return get_agent_registry().get(slug)


def list_agents() -> list[AgentDef]:
    """List all available agents, sorted by name."""
    return sorted(get_agent_registry().values(), key=lambda a: a.name)


def reload_agents() -> None:
    """Force reload the agent registry (e.g., after adding custom agents)."""
    global _registry
    _registry = None
    get_agent_registry()


__all__ = ["AgentDef", "get_agent", "list_agents", "get_agent_registry", "reload_agents"]
