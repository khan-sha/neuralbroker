"""
Google Vertex AI provider adapter.

Uses google-cloud-aiplatform SDK to call Vertex AI Gemini models.
Translates OpenAI messages to Vertex Content objects and back.
Requires: pip install google-cloud-aiplatform
"""
import json
import os
import time
from typing import AsyncIterator

from neuralbrok.providers.base import (
    BaseProvider,
    ProviderError,
    BackendUnavailableError,
)

_DEFAULT_LOCATION = "us-central1"


class VertexProvider(BaseProvider):
    """Adapter for Google Vertex AI (Gemini models).

    Uses the google-cloud-aiplatform SDK. Authentication is via
    GOOGLE_APPLICATION_CREDENTIALS service account JSON.
    If the SDK is not installed, raises ProviderError on first chat() call.
    """

    SUPPORTED_MODELS = [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
    ]

    def __init__(self, name: str):
        super().__init__(name=name, provider_type="cloud")
        self.project = os.getenv("VERTEX_PROJECT", "")
        self.location = os.getenv("VERTEX_LOCATION", _DEFAULT_LOCATION)

    def _load_sdk(self):
        """Load Vertex AI SDK or raise ProviderError."""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel, Content, Part
            return vertexai, GenerativeModel, Content, Part
        except ImportError:
            raise ProviderError(
                self.name,
                "google-cloud-aiplatform not installed: "
                "pip install google-cloud-aiplatform",
            )

    def _build_contents(self, messages: list[dict], Content, Part):
        """Convert OpenAI messages to Vertex Content objects."""
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "system":
                continue  # System prompts handled separately if needed
            vertex_role = "model" if role == "assistant" else "user"
            contents.append(Content(role=vertex_role, parts=[Part.from_text(text)]))
        return contents

    async def chat(
        self, payload: dict, stream: bool = True
    ) -> AsyncIterator[str]:
        """Forward request to Vertex AI and yield OpenAI-compatible SSE."""
        vertexai, GenerativeModel, Content, Part = self._load_sdk()

        model_name = payload.get("model", "gemini-1.5-flash")
        chunk_id = f"chatcmpl-nb-{self.name}-{int(time.time())}"
        contents = self._build_contents(
            payload.get("messages", []), Content, Part
        )

        try:
            vertexai.init(project=self.project, location=self.location)
            model = GenerativeModel(model_name)

            if stream:
                responses = await model.generate_content_async(
                    contents, stream=True
                )
                async for response in responses:
                    text = response.text if hasattr(response, "text") else ""
                    openai_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(openai_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            else:
                response = await model.generate_content_async(
                    contents, stream=False
                )
                text = response.text if hasattr(response, "text") else ""
                openai_response = {
                    "id": chunk_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield json.dumps(openai_response)

        except ProviderError:
            raise
        except Exception as e:
            err_str = str(e)
            if "UNAVAILABLE" in err_str or "Connection" in err_str:
                raise BackendUnavailableError(self.name, err_str)
            raise ProviderError(self.name, err_str)
