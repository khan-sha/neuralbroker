"""
Groq provider adapter.

OpenAI-compatible — inherits from the shared base.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    """Adapter for Groq cloud inference API.

    Groq speaks native OpenAI format. No request transformation needed.
    Known for extremely low latency (LPU inference).
    """

    SUPPORTED_MODELS = [
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma-7b-it",
    ]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
