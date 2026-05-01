"""
ReasoningBank: The self-learning module that stores successful swarm task trajectories.
Equivalent to NeuralBroker's SONA pattern memory.
"""
import logging
from neuralbrok.memory.vector_store import AgentDB

logger = logging.getLogger(__name__)

class ReasoningBank:
    def __init__(self):
        self.db = AgentDB(collection_name="reasoning_bank")

    def store_trajectory(self, task: str, plan: str, outcome: str):
        """Stores a successful execution path for agents to learn from."""
        text = f"Task: {task}\nPlan: {plan}\nOutcome: {outcome}"
        metadata = {"type": "trajectory"}
        self.db.add(text, metadata)
        logger.info("Successfully stored task trajectory into ReasoningBank.")

    def retrieve_similar_lessons(self, current_task: str) -> str:
        """Retrieves past lessons that are semantically similar to the current task."""
        results = self.db.search(current_task, limit=2)
        if not results:
            return ""
            
        lessons = []
        for doc, score in results:
            if score > 0.5:  # Relevance threshold
                lessons.append(doc["text"])
                
        if lessons:
            return "Prior learnings from similar tasks:\n" + "\n---\n".join(lessons)
        return ""
