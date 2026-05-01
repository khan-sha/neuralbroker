"""
PII Redactor: Security module to strip sensitive information before routing to untrusted backends.
Equivalent to NeuralBroker's AIDefence privacy filter.
"""
import re

class PIIRedactor:
    def __init__(self):
        self.rules = [
            # Basic Email Regex
            (re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'), '[REDACTED_EMAIL]'),
            # AWS / API Key heuristics
            (re.compile(r'(?i)(api[_-]?key|secret|token|password)[\s:=]+[\'"]?[a-zA-Z0-9\-_]{16,}[\'"]?'), '[REDACTED_SECRET]'),
            # SSN / Credit Card patterns (simplified)
            (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[REDACTED_SSN]'),
            (re.compile(r'\b(?:\d{4}[ -]?){3}\d{4}\b'), '[REDACTED_CC]')
        ]

    def sanitize(self, text: str) -> str:
        """Scan and redact PII from outbound messages."""
        if not text:
            return text
            
        sanitized = text
        for pattern, replacement in self.rules:
            sanitized = pattern.sub(replacement, sanitized)
            
        return sanitized
