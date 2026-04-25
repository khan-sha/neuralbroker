"""
NeuralBroker integrations for external tools and frameworks.

Beta features:
- claude_code: Claude Code terminal connection with routing context
"""

from .claude_code import ClaudeCodeTerminal, launch_code_with_routing_context
from .agents import AgentIntegration, AGENT_REGISTRY, setup, list_agents, check_status, remove_agent, get_installed_integrations, _nb_url

__all__ = [
    "ClaudeCodeTerminal", 
    "launch_code_with_routing_context",
    "AgentIntegration",
    "AGENT_REGISTRY",
    "setup",
    "list_agents",
    "check_status",
    "remove_agent",
    "get_installed_integrations",
    "_nb_url"
]
