"""
Lepton AI provider adapter.

OpenAI-compatible — inherits from the shared base.
"""
from neuralbrok.providers._openai_compat import OpenAICompatibleProvider


class LeptonProvider(OpenAICompatibleProvider):
    """Adapter for Lepton AI inference API.

    Lepton speaks native OpenAI format. Serverless GPU cloud
    with pay-per-use pricing.
    """

    SUPPORTED_MODELS = ["llama3-1-8b", "llama3-1-70b", "mistral-7b"]

    def __init__(self, name: str, base_url: str, api_key: str):
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            provider_type="cloud",
        )
