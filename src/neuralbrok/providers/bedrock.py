"""
AWS Bedrock provider adapter.

Uses boto3 to call Amazon Bedrock Runtime. boto3 is synchronous,
so calls are wrapped with asyncio.get_event_loop().run_in_executor().
"""
import asyncio
import json
import time
from typing import AsyncIterator

from neuralbrok.providers.base import (
    BaseProvider,
    ProviderError,
    RateLimitError,
    BackendUnavailableError,
)

import os

_DEFAULT_REGION = "us-east-1"


class BedrockProvider(BaseProvider):
    """Adapter for Amazon Bedrock Runtime.

    Wraps the synchronous boto3 SDK with asyncio executors.
    Supports Anthropic Claude and Meta Llama models on Bedrock.
    Requires: pip install boto3
    """

    SUPPORTED_MODELS = [
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "meta.llama3-8b-instruct-v1:0",
    ]

    def __init__(self, name: str):
        super().__init__(name=name, provider_type="cloud")
        self.aws_region = os.getenv("AWS_REGION", _DEFAULT_REGION)
        self._client = None

    def _get_client(self):
        """Lazily create boto3 client."""
        try:
            import boto3
        except ImportError:
            raise ProviderError(
                self.name, "boto3 not installed: pip install boto3"
            )
        if self._client is None:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.aws_region,
            )
        return self._client

    def _build_body(self, payload: dict, stream: bool) -> dict:
        """Build Bedrock request body for Anthropic models."""
        messages = payload.get("messages", [])
        # Filter out system messages for content
        filtered = [m for m in messages if m.get("role") != "system"]
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": payload.get("max_tokens", 2048),
            "messages": filtered,
        }

    async def chat(
        self, payload: dict, stream: bool = True
    ) -> AsyncIterator[str]:
        """Forward request to Bedrock Runtime."""
        model = payload.get("model", "anthropic.claude-3-haiku-20240307-v1:0")
        chunk_id = f"chatcmpl-nb-{self.name}-{int(time.time())}"
        body = self._build_body(payload, stream)
        body_bytes = json.dumps(body).encode("utf-8")

        loop = asyncio.get_event_loop()

        try:
            client = self._get_client()
        except ProviderError:
            raise

        try:
            if stream:
                def _invoke_stream():
                    return client.invoke_model_with_response_stream(
                        modelId=model,
                        body=body_bytes,
                        contentType="application/json",
                        accept="application/json",
                    )

                response = await loop.run_in_executor(None, _invoke_stream)
                event_stream = response["body"]

                def _read_stream():
                    chunks = []
                    for event in event_stream:
                        chunk_data = event.get("chunk", {})
                        raw = chunk_data.get("bytes", b"")
                        if raw:
                            try:
                                parsed = json.loads(raw)
                                chunks.append(parsed)
                            except json.JSONDecodeError:
                                pass
                    return chunks

                events = await loop.run_in_executor(None, _read_stream)

                for event in events:
                    event_type = event.get("type", "")
                    if event_type == "content_block_delta":
                        text = event.get("delta", {}).get("text", "")
                        openai_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(openai_chunk)}\n\n"
                    elif event_type == "message_stop":
                        break

                yield "data: [DONE]\n\n"

            else:
                def _invoke():
                    return client.invoke_model(
                        modelId=model,
                        body=body_bytes,
                        contentType="application/json",
                        accept="application/json",
                    )

                response = await loop.run_in_executor(None, _invoke)
                data = json.loads(response["body"].read())

                # Anthropic on Bedrock returns same content shape
                content_blocks = data.get("content", [])
                text = "".join(
                    block.get("text", "")
                    for block in content_blocks
                    if block.get("type") == "text"
                )

                usage = data.get("usage", {})
                openai_response = {
                    "id": chunk_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
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
                    "usage": {
                        "prompt_tokens": usage.get("input_tokens", 0),
                        "completion_tokens": usage.get("output_tokens", 0),
                        "total_tokens": (
                            usage.get("input_tokens", 0)
                            + usage.get("output_tokens", 0)
                        ),
                    },
                }
                yield json.dumps(openai_response)

        except ProviderError:
            raise
        except Exception as e:
            # Map boto3 exceptions
            err_str = str(e)
            if "ThrottlingException" in err_str or "TooManyRequests" in err_str:
                raise RateLimitError(self.name)
            if "EndpointResolutionError" in err_str or "ConnectTimeoutError" in err_str:
                raise BackendUnavailableError(self.name, err_str)
            raise ProviderError(self.name, err_str)
