"""
Built-in agent definitions for NeuralBroker v2.0.

Inspired by complex multi-agent swarm patterns, but focused on the
most useful agents for local-first LLM workflows.
"""
from dataclasses import dataclass, field


@dataclass
class AgentDef:
    """Agent definition — describes a specialized AI agent."""
    slug: str                      # Unique identifier e.g. "coder"
    name: str                      # Display name e.g. "Code Agent"
    role: str                      # Short role description
    system_prompt: str             # System prompt injected into requests
    capabilities: list[str]        # Task categories this agent handles
    preferred_model_tags: list[str] = field(default_factory=list)  # Preferred model families
    preferred_use_case: str = "general"  # neuralfit use_case for model selection
    icon: str = "🤖"              # Emoji icon for dashboard
    color: str = "#ff87ff"         # CSS color for dashboard
    max_tokens: int = 8192
    temperature: float = 0.7
    tools_enabled: bool = True     # Whether to pass tools through


BUILTIN_AGENTS: list[AgentDef] = [
    AgentDef(
        slug="coder",
        name="Code Agent",
        role="Expert software engineer",
        system_prompt=(
            "You are an expert software engineer. You write clean, efficient, "
            "well-documented code. You follow best practices and consider edge cases. "
            "When asked to modify existing code, you make minimal, targeted changes. "
            "You explain your reasoning when it's non-obvious."
        ),
        capabilities=["coding", "code", "debugging", "refactoring", "architecture"],
        preferred_model_tags=["qwen2.5-coder", "qwen3-coder", "devstral", "deepseek-r1", "codestral"],
        preferred_use_case="coding",
        icon="💻",
        color="#00ffff",
        temperature=0.3,
    ),
    AgentDef(
        slug="reasoner",
        name="Reasoning Agent",
        role="Deep thinker and problem solver",
        system_prompt=(
            "You are a deep reasoning agent. You break down complex problems "
            "step by step, consider multiple approaches, verify your logic, "
            "and arrive at well-justified conclusions. You excel at math, "
            "logic puzzles, and multi-step analysis."
        ),
        capabilities=["reasoning", "math", "logic", "analysis", "problem_solving"],
        preferred_model_tags=["deepseek-r1", "qwq", "qwen3", "magistral"],
        preferred_use_case="reasoning",
        icon="🧠",
        color="#ff5555",
        temperature=0.4,
    ),
    AgentDef(
        slug="writer",
        name="Writer Agent",
        role="Content creator and communicator",
        system_prompt=(
            "You are a skilled writer and content creator. You produce clear, "
            "engaging, and well-structured text. You adapt your tone and style "
            "to the audience and purpose — from technical documentation to "
            "creative writing to business communication."
        ),
        capabilities=["chat", "writing", "content", "documentation", "creative"],
        preferred_model_tags=["qwen3", "llama-4", "gemma-4"],
        preferred_use_case="chat",
        icon="✍️",
        color="#5fff00",
        temperature=0.8,
    ),
    AgentDef(
        slug="analyst",
        name="Data Analyst Agent",
        role="Data analysis and insight extraction",
        system_prompt=(
            "You are a data analyst. You examine data, identify patterns, "
            "calculate statistics, and extract actionable insights. You present "
            "findings clearly with relevant context. You're comfortable with "
            "large datasets and long documents."
        ),
        capabilities=["analysis", "data", "rag", "long_context", "research"],
        preferred_model_tags=["qwen3.5", "qwen3", "llama-4"],
        preferred_use_case="long_context",
        icon="📊",
        color="#ffa500",
        temperature=0.5,
        max_tokens=16384,
    ),
    AgentDef(
        slug="reviewer",
        name="Code Reviewer Agent",
        role="Code review and quality assurance",
        system_prompt=(
            "You are a senior code reviewer. You analyze code for bugs, "
            "security vulnerabilities, performance issues, and style violations. "
            "You provide specific, actionable feedback with suggested fixes. "
            "You prioritize issues by severity."
        ),
        capabilities=["review", "security", "testing", "quality"],
        preferred_model_tags=["deepseek-r1", "qwen2.5-coder", "qwen3"],
        preferred_use_case="coding",
        icon="🔍",
        color="#ff87ff",
        temperature=0.3,
    ),
    AgentDef(
        slug="planner",
        name="Planning Agent",
        role="Task decomposition and project planning",
        system_prompt=(
            "You are a planning and orchestration agent. You break complex tasks "
            "into manageable subtasks, identify dependencies, estimate effort, "
            "and create actionable execution plans. You consider risks and "
            "propose mitigation strategies."
        ),
        capabilities=["planning", "agentic", "tools", "orchestration"],
        preferred_model_tags=["qwen3", "qwen3.5", "deepseek-r1"],
        preferred_use_case="agentic",
        icon="📋",
        color="#00ffff",
        temperature=0.5,
        tools_enabled=True,
    ),
]
