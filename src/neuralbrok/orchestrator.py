"""
Agent orchestration engine — Swarm-native.

Implements:
- TaskClassifier: Classifies incoming tasks into categories
- AgentRouter: Maps task categories → best agent → best model
- SwarmCoordinator: Multi-agent workflow orchestration

Architecture:
    User Request → Classifier → Router → Agent Selection → Execution → Result
                                             ↓
                                 [coder, reasoner, writer,
                                  analyst, reviewer, planner]
"""
import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import httpx

from neuralbrok.agents import get_agent, list_agents, AgentDef
from neuralbrok.llmfit_scorer import (
    SystemSpecs,
    rank_models,
    detect_system_specs,
    model_fit_to_dict,
)

logger = logging.getLogger(__name__)


# ── Task Classification ──────────────────────────────────────────────────────

class TaskCategory(str, Enum):
    CODING      = "coding"
    REASONING   = "reasoning"
    WRITING     = "writing"
    ANALYSIS    = "analysis"
    REVIEW      = "review"
    PLANNING    = "planning"
    CHAT        = "chat"
    MATH        = "math"
    TOOLS       = "tools"


# Keyword-based classifier (fast, no LLM needed)
_CATEGORY_KEYWORDS: dict[TaskCategory, list[str]] = {
    TaskCategory.CODING: [
        "code", "implement", "function", "class", "bug", "fix", "error",
        "program", "script", "api", "endpoint", "debug", "refactor",
        "test", "unittest", "compile", "build", "syntax", "variable",
        "import", "module", "package", "library", "framework", "git",
        "commit", "merge", "pull request", "pr", "deploy", "dockerfile",
    ],
    TaskCategory.REASONING: [
        "think", "reason", "explain why", "analyze", "consider",
        "evaluate", "compare", "contrast", "pros and cons", "tradeoff",
        "deduce", "infer", "conclude", "hypothesis", "theory",
    ],
    TaskCategory.MATH: [
        "calculate", "equation", "formula", "math", "integral",
        "derivative", "probability", "statistics", "matrix", "vector",
        "algebra", "geometry", "proof", "theorem", "compute",
    ],
    TaskCategory.WRITING: [
        "write", "draft", "compose", "essay", "article", "blog",
        "email", "letter", "story", "poem", "documentation", "docs",
        "readme", "summary", "describe", "narrative", "creative",
    ],
    TaskCategory.ANALYSIS: [
        "data", "dataset", "analyze", "insight", "trend", "pattern",
        "report", "metric", "dashboard", "chart", "graph", "csv",
        "json", "parse", "extract", "aggregate", "correlation",
    ],
    TaskCategory.REVIEW: [
        "review", "audit", "check", "verify", "validate", "security",
        "vulnerability", "performance", "optimize", "lint", "style",
        "best practice", "code review", "feedback",
    ],
    TaskCategory.PLANNING: [
        "plan", "design", "architect", "roadmap", "milestone",
        "timeline", "schedule", "breakdown", "subtask", "workflow",
        "strategy", "prioritize", "scope", "requirement", "spec",
    ],
    TaskCategory.TOOLS: [
        "search", "browse", "fetch", "download", "install",
        "run", "execute", "shell", "command", "terminal",
    ],
}

# Category → Agent slug mapping
_CATEGORY_AGENT: dict[TaskCategory, str] = {
    TaskCategory.CODING:    "coder",
    TaskCategory.REASONING: "reasoner",
    TaskCategory.MATH:      "reasoner",
    TaskCategory.WRITING:   "writer",
    TaskCategory.ANALYSIS:  "analyst",
    TaskCategory.REVIEW:    "reviewer",
    TaskCategory.PLANNING:  "planner",
    TaskCategory.CHAT:      "writer",
    TaskCategory.TOOLS:     "planner",
}


class TaskClassifier:
    """Classifies tasks using keyword matching + optional LLM classification."""

    def classify_fast(self, text: str) -> list[TaskCategory]:
        """Fast keyword-based classification (< 1ms, no LLM call)."""
        text_lower = text.lower()
        scores: dict[TaskCategory, int] = {}

        for category, keywords in _CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[category] = score

        if not scores:
            return [TaskCategory.CHAT]

        # Return categories sorted by keyword match count
        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [cat for cat, _ in sorted_cats[:3]]

    async def classify_llm(
        self, text: str, model: str = "phi-4-mini", host: str = "localhost:11434"
    ) -> list[TaskCategory]:
        """LLM-based classification (more accurate, ~200ms with small model).

        Falls back to keyword-based if LLM is unavailable.
        """
        system = (
            "Classify this task into categories. Respond with ONLY a JSON array of strings.\n"
            'Categories: ["coding", "reasoning", "math", "writing", "analysis", "review", "planning", "chat", "tools"]\n'
            "Return at most 3 categories, most relevant first."
        )
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.post(
                    f"http://{host}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": text[:300]},
                        ],
                        "stream": False,
                    },
                )
                if resp.status_code == 200:
                    content = resp.json()["message"]["content"]
                    match = re.search(r'\[.*\]', content, re.DOTALL)
                    if match:
                        cats = json.loads(match.group(0))
                        return [TaskCategory(c) for c in cats if c in TaskCategory.__members__.values()]
        except Exception:
            pass

        return self.classify_fast(text)


# ── Agent Router ─────────────────────────────────────────────────────────────

@dataclass
class AgentDecision:
    """Result of the agent routing decision."""
    agent: AgentDef
    categories: list[str]
    recommended_model: str
    classification_method: str  # "keyword" or "llm"
    classification_ms: float
    reason: str


class AgentRouter:
    """Routes tasks to the best agent based on classification and hardware."""

    def __init__(self, hw_specs: Optional[SystemSpecs] = None):
        self.classifier = TaskClassifier()
        self._hw_specs = hw_specs
        self._cached_specs: Optional[SystemSpecs] = None

    @property
    def hw_specs(self) -> SystemSpecs:
        if self._hw_specs:
            return self._hw_specs
        if self._cached_specs is None:
            self._cached_specs = detect_system_specs()
        return self._cached_specs

    def route_fast(self, task_text: str) -> AgentDecision:
        """Fast routing using keyword classification (< 1ms)."""
        start = time.perf_counter()
        categories = self.classifier.classify_fast(task_text)
        class_ms = (time.perf_counter() - start) * 1000

        primary = categories[0]
        agent_slug = _CATEGORY_AGENT.get(primary, "writer")
        agent = get_agent(agent_slug)

        if agent is None:
            # Fallback to writer if agent not found
            agent = get_agent("writer")

        # Find best model for this agent's use case
        recommended = self._pick_model(agent)

        return AgentDecision(
            agent=agent,
            categories=[c.value for c in categories],
            recommended_model=recommended,
            classification_method="keyword",
            classification_ms=round(class_ms, 2),
            reason=f"keyword:{primary.value}→{agent.slug}",
        )

    async def route(self, task_text: str, use_llm: bool = False) -> AgentDecision:
        """Route a task to the best agent.

        Args:
            task_text: The task/prompt to classify
            use_llm: Whether to use LLM for classification (slower but more accurate)
        """
        start = time.perf_counter()

        if use_llm:
            categories = await self.classifier.classify_llm(task_text)
            method = "llm"
        else:
            categories = self.classifier.classify_fast(task_text)
            method = "keyword"

        class_ms = (time.perf_counter() - start) * 1000

        primary = categories[0]
        agent_slug = _CATEGORY_AGENT.get(primary, "writer")
        agent = get_agent(agent_slug)

        if agent is None:
            agent = get_agent("writer")

        recommended = self._pick_model(agent)

        return AgentDecision(
            agent=agent,
            categories=[c.value for c in categories],
            recommended_model=recommended,
            classification_method=method,
            classification_ms=round(class_ms, 2),
            reason=f"{method}:{primary.value}→{agent.slug}",
        )

    def _pick_model(self, agent: AgentDef) -> str:
        """Pick the best model for an agent using llmfit scoring."""
        try:
            fits = rank_models(
                hw=self.hw_specs,
                use_case=agent.preferred_use_case,
                max_results=3,
            )
            if fits:
                # Prefer installed models, then highest composite score
                installed = [f for f in fits if f.is_installed]
                if installed:
                    return installed[0].ollama_tag
                return fits[0].ollama_tag
        except Exception as e:
            logger.warning(f"Model selection failed for agent {agent.slug}: {e}")

        # Fallback to first preferred model tag
        if agent.preferred_model_tags:
            return agent.preferred_model_tags[0]
        return "qwen3:8b"


# ── Swarm Coordinator ────────────────────────────────────────────────────────

class SwarmStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    COMPLETE = "complete"
    FAILED   = "failed"


@dataclass
class SwarmTask:
    """A subtask within a swarm execution."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    agent_slug: str = ""
    model: str = ""
    status: SwarmStatus = SwarmStatus.PENDING
    result: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: str = ""


@dataclass
class Swarm:
    """A multi-agent swarm executing a complex task."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    objective: str = ""
    tasks: list[SwarmTask] = field(default_factory=list)
    status: SwarmStatus = SwarmStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    final_result: str = ""


class SwarmCoordinator:
    """Coordinates multi-agent swarm workflows.

    Implements the NeuralBroker Plan → Execute → Review pattern:
    1. Planner agent decomposes the task
    2. Specialized agents execute subtasks
    3. Reviewer agent validates results
    """

    def __init__(self, router: AgentRouter):
        self.router = router
        self._swarms: dict[str, Swarm] = {}

    def create_swarm(self, objective: str) -> Swarm:
        """Create a new swarm for a complex objective."""
        swarm = Swarm(objective=objective)
        self._swarms[swarm.id] = swarm
        return swarm

    def get_swarm(self, swarm_id: str) -> Optional[Swarm]:
        """Get a swarm by ID."""
        return self._swarms.get(swarm_id)

    def list_swarms(self) -> list[Swarm]:
        """List all swarms."""
        return list(self._swarms.values())

    async def decompose(self, swarm: Swarm) -> list[SwarmTask]:
        """Use the planner agent to decompose the objective into subtasks.

        This is a lightweight decomposition — returns pre-defined task
        structure for common patterns, or uses LLM for custom decomposition.
        """
        # Simple decomposition: classify objective and create 3-phase pipeline
        decision = self.router.route_fast(swarm.objective)

        tasks = [
            SwarmTask(
                description=f"Plan: {swarm.objective}",
                agent_slug="planner",
            ),
            SwarmTask(
                description=f"Execute: {swarm.objective}",
                agent_slug=decision.agent.slug,
            ),
            SwarmTask(
                description=f"Review the execution result for: {swarm.objective}",
                agent_slug="reviewer",
            ),
        ]

        # Assign models to each task
        for task in tasks:
            agent = get_agent(task.agent_slug)
            if agent:
                task.model = self.router._pick_model(agent)

        swarm.tasks = tasks
        return tasks

    async def execute_swarm(
        self,
        swarm: Swarm,
        ollama_host: str = "localhost:11434",
    ) -> Swarm:
        """Execute all tasks in the swarm sequentially.

        Each task's result is passed as context to the next task.
        """
        swarm.status = SwarmStatus.RUNNING
        context = f"Objective: {swarm.objective}\n"

        if not swarm.tasks:
            await self.decompose(swarm)

        for task in swarm.tasks:
            task.status = SwarmStatus.RUNNING
            task.started_at = datetime.now(timezone.utc)

            agent = get_agent(task.agent_slug)
            if not agent:
                task.status = SwarmStatus.FAILED
                task.error = f"Agent '{task.agent_slug}' not found"
                continue

            try:
                result = await self._execute_task(
                    task=task,
                    agent=agent,
                    context=context,
                    host=ollama_host,
                )
                task.result = result
                task.status = SwarmStatus.COMPLETE
                task.completed_at = datetime.now(timezone.utc)

                # Add result to context for next task
                context += f"\n[{agent.name} output]:\n{result}\n"

            except Exception as e:
                task.status = SwarmStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now(timezone.utc)
                logger.error(f"Swarm task {task.id} failed: {e}")

        # Determine overall swarm status
        if all(t.status == SwarmStatus.COMPLETE for t in swarm.tasks):
            swarm.status = SwarmStatus.COMPLETE
            swarm.final_result = swarm.tasks[-1].result if swarm.tasks else ""
        else:
            swarm.status = SwarmStatus.FAILED
            swarm.final_result = "Some tasks failed. See individual task results."

        swarm.completed_at = datetime.now(timezone.utc)
        return swarm

    async def _execute_task(
        self,
        task: SwarmTask,
        agent: AgentDef,
        context: str,
        host: str,
    ) -> str:
        """Execute a single task via Ollama."""
        model = task.model or "qwen3:8b"

        messages = [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": f"{context}\n\nTask: {task.description}"},
        ]

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"http://{host}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                    },
                )
                if resp.status_code == 200:
                    return resp.json()["message"]["content"]
                else:
                    raise RuntimeError(f"Ollama returned {resp.status_code}")
        except Exception as e:
            raise RuntimeError(f"Task execution failed: {e}")


# ── Convenience Functions ────────────────────────────────────────────────────

def swarm_to_dict(swarm: Swarm) -> dict:
    """Convert Swarm to JSON-serializable dict."""
    return {
        "id": swarm.id,
        "objective": swarm.objective,
        "status": swarm.status.value,
        "created_at": swarm.created_at.isoformat(),
        "completed_at": swarm.completed_at.isoformat() if swarm.completed_at else None,
        "final_result": swarm.final_result,
        "tasks": [
            {
                "id": t.id,
                "description": t.description,
                "agent_slug": t.agent_slug,
                "model": t.model,
                "status": t.status.value,
                "result": t.result[:500] if t.result else "",
                "error": t.error,
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
            }
            for t in swarm.tasks
        ],
    }


def agent_decision_to_dict(decision: AgentDecision) -> dict:
    """Convert AgentDecision to JSON-serializable dict."""
    return {
        "agent": {
            "slug": decision.agent.slug,
            "name": decision.agent.name,
            "role": decision.agent.role,
            "icon": decision.agent.icon,
            "color": decision.agent.color,
            "capabilities": decision.agent.capabilities,
        },
        "categories": decision.categories,
        "recommended_model": decision.recommended_model,
        "classification_method": decision.classification_method,
        "classification_ms": decision.classification_ms,
        "reason": decision.reason,
    }
