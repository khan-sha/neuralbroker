"""
Kimi (Moonshot AI) provider adapter.

OpenAI-compatible — inherits from the shared base.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class KimiProvider(OpenAICompatibleProvider):
    """Adapter for Kimi (Moonshot AI) inference API.

    Kimi speaks native OpenAI format via the Moonshot API.
    Strong 1M token context window — ideal for long-document tasks.
    """

    SUPPORTED_MODELS = [
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k",
    ]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
