"""
Zhipu AI provider adapter.

OpenAI-compatible — inherits from the shared base.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class ZhipuProvider(OpenAICompatibleProvider):
    """Adapter for Zhipu AI (BigModel) inference API.

    Zhipu speaks native OpenAI format via their paas/v4 endpoint.
    Models are GLM-4 family — strong Chinese and code capabilities.
    """

    SUPPORTED_MODELS = ["glm-4", "glm-4-flash", "glm-4-air", "glm-3-turbo"]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
