"""
DeepSeek provider adapter.

OpenAI-compatible — inherits from the shared base.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """Adapter for DeepSeek inference API.

    DeepSeek speaks native OpenAI format.
    Extremely cost-effective, strong on coding tasks.
    """

    SUPPORTED_MODELS = ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
