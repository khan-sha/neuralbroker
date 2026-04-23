"""
Yi (01.AI) provider adapter.

OpenAI-compatible — inherits from the shared base.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class YiProvider(OpenAICompatibleProvider):
    """Adapter for Yi (01.AI) inference API.

    Yi speaks native OpenAI format. Strong reasoning and
    multilingual capabilities from 01.AI.
    """

    SUPPORTED_MODELS = ["yi-lightning", "yi-large", "yi-medium"]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
