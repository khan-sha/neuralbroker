"""
Baichuan AI provider adapter.

OpenAI-compatible — inherits from the shared base.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class BaichuanProvider(OpenAICompatibleProvider):
    """Adapter for Baichuan AI inference API.

    Baichuan speaks native OpenAI format. Strong Chinese language
    understanding from Baichuan Intelligence.
    """

    SUPPORTED_MODELS = ["Baichuan4", "Baichuan3-Turbo", "Baichuan2-Turbo"]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
