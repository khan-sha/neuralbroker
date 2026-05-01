"""
Injection Shield: Heuristic engine to detect prompt injection and jailbreaks.
Equivalent to Ruflo's AIDefence.
"""
import re

class InjectionShield:
    def __init__(self):
        self.jailbreak_signatures = [
            "ignore previous instructions",
            "disregard all previous",
            "you are now a",
            "system prompt:",
            "as an AI language model, you are forbidden",
            "from now on, you will",
            "DAN", "jailbreak"
        ]

    def is_safe(self, prompt: str) -> bool:
        """Returns True if the prompt appears safe from injection attempts."""
        if not prompt:
            return True
            
        prompt_lower = prompt.lower()
        for signature in self.jailbreak_signatures:
            if signature in prompt_lower:
                return False
                
        # Heuristic: Extremely high number of special symbols often indicates adversarial encoding
        symbol_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', prompt)) / max(len(prompt), 1)
        if symbol_ratio > 0.4:
            return False
            
        return True
