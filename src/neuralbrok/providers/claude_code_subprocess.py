"""
Claude Code Subprocess Provider.

Routes requests through the installed `claude` CLI using its OAuth session.
No API key required — uses Claude Pro/Max subscription from ~/.claude/.credentials.json.
"""
import asyncio
import json
import shutil
import time
from typing import AsyncIterator

from neuralbrok.providers.base import (
    BaseProvider,
    BackendUnavailableError,
    ProviderError,
)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_MODEL_ALIASES = {
    "claude-sonnet-4-5": "sonnet",
    "claude-opus-4-5": "opus",
    "claude-haiku-3-5": "haiku",
    "claude-3-5-sonnet-20241022": "sonnet",
    "claude-3-haiku-20240307": "haiku",
    "claude-sonnet-4-6": "sonnet",
    "claude-opus-4-7": "opus",
    "claude-haiku-4-5-20251001": "haiku",
    "sonnet": "sonnet",
    "opus": "opus",
    "haiku": "haiku",
}

SUPPORTED_MODELS = [
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-haiku-4-5-20251001",
    "sonnet",
    "opus",
    "haiku",
]


def _messages_to_prompt(messages: list[dict]) -> tuple[str, str | None]:
    """Convert OpenAI messages list to flat prompt + optional system string."""
    system = None
    history = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        if role == "system":
            system = content
        elif role == "user":
            history.append(f"Human: {content}")
        elif role == "assistant":
            history.append(f"Assistant: {content}")

    if not history:
        return "", system
    if len(history) == 1 and history[0].startswith("Human: "):
        return history[0][len("Human: "):], system
    return "\n\n".join(history), system


class ClaudeCodeSubprocessProvider(BaseProvider):
    """Runs inference via installed `claude` CLI using Claude Pro/Max OAuth.

    No API key needed. Shells out to `claude -p <prompt> --output-format json`.
    Claude Code CLI uses the OAuth session from ~/.claude/.credentials.json.
    Cost = $0 at margin (subscription already paid).
    """

    SUPPORTED_MODELS = SUPPORTED_MODELS

    def __init__(self, name: str = "claude_code", model: str = _DEFAULT_MODEL):
        super().__init__(name=name, provider_type="cloud")
        self.model = _MODEL_ALIASES.get(model, model)
        self._claude_path = shutil.which("claude")

    async def chat(self, payload: dict, stream: bool = True) -> AsyncIterator[str]:
        if not self._claude_path:
            raise BackendUnavailableError(self.name, "claude CLI not found in PATH")

        messages = payload.get("messages", [])
        raw_model = payload.get("model", self.model)
        model = _MODEL_ALIASES.get(raw_model, self.model)
        chunk_id = f"chatcmpl-nb-{self.name}-{int(time.time())}"

        prompt, system = _messages_to_prompt(messages)
        if not prompt:
            raise ProviderError(self.name, "Empty prompt", retryable=False)

        if stream:
            async for chunk in self._stream(prompt, system, model, chunk_id):
                yield chunk
        else:
            result = await self._run_sync(prompt, system, model)
            yield json.dumps({
                "id": chunk_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": f"claude-{model}" if not model.startswith("claude") else model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

    async def _run_sync(self, prompt: str, system: str | None, model: str) -> str:
        cmd = [
            self._claude_path, "-p", prompt,
            "--output-format", "json",
            "--model", model,
        ]
        if system:
            cmd += ["--system-prompt", system]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
        except asyncio.TimeoutError:
            proc.kill()
            raise ProviderError(self.name, "claude CLI timeout after 120s")

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            raise ProviderError(self.name, f"claude exit {proc.returncode}: {err}")

        raw = stdout.decode(errors="replace").strip()
        try:
            return json.loads(raw).get("result", raw)
        except json.JSONDecodeError:
            return raw

    async def _stream(
        self, prompt: str, system: str | None, model: str, chunk_id: str
    ) -> AsyncIterator[str]:
        cmd = [
            self._claude_path, "-p", prompt,
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--model", model,
        ]
        if system:
            cmd += ["--system-prompt", system]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        yielded = False
        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")
            if etype == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text" and block.get("text"):
                        openai_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": block["text"]},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(openai_chunk)}\n\n"
                        yielded = True

            elif etype == "result":
                if not yielded:
                    t = event.get("result", "")
                    if t:
                        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': t}, 'finish_reason': None}]})}\n\n"

        await proc.wait()
        yield "data: [DONE]\n\n"

    async def health_check(self) -> bool:
        if not self._claude_path:
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                self._claude_path, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return proc.returncode == 0
        except Exception:
            return False
