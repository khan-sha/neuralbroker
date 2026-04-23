"""
Qwen (Alibaba DashScope) provider adapter.

OpenAI-compatible — inherits from the shared base.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class QwenProvider(OpenAICompatibleProvider):
    """Adapter for Qwen inference API via DashScope.

    Via DashScope OpenAI-compatible mode.
    Alibaba's Qwen model family, strong multilingual performance.
    """

    SUPPORTED_MODELS = [
        "qwen-turbo",
        "qwen-plus",
        "qwen-max",
        "qwen2.5-72b-instruct",
    ]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
