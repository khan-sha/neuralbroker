"""
Fireworks AI provider adapter.

OpenAI-compatible — inherits from the shared base.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class FireworksProvider(OpenAICompatibleProvider):
    """Adapter for Fireworks AI inference API.

    Fireworks speaks native OpenAI format. Fast inference with compound
    AI system support and function calling.
    """

    SUPPORTED_MODELS = [
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "accounts/fireworks/models/mixtral-8x22b-instruct",
    ]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
