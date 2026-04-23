"""
AI21 Labs provider adapter.

OpenAI-compatible — inherits from the shared base.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class AI21Provider(OpenAICompatibleProvider):
    """Adapter for AI21 Labs inference API.

    AI21 speaks native OpenAI format via their Studio API.
    Features the Jamba model family (SSM-Transformer hybrid).
    """

    SUPPORTED_MODELS = ["jamba-1.5-mini", "jamba-1.5-large", "j2-ultra", "j2-mid"]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
