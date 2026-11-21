"""
Talks to any OpenAI-compatible chat server: llama.cpp's llama-server on the laptop now, a GPU server later. Changing
server is a settings change (CHIRII_LLM_BASE_URL), not a code change.

Settings for every call are fixed for reproducibility: temperature 0, at most 256 new tokens (CRITERIA.md).
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import httpx

from chirii.app.logging_setup import log_event


class LLMError(RuntimeError):
    """The model server could not be reached or gave an unusable reply."""


@dataclass
class LLMReply:
    text: str
    tokens_in: int
    tokens_out: int
    duration_ms: int


class ChatModel(Protocol):
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 256, json_schema: Optional[Dict[str, Any]] = None,
             purpose: str = "chat") -> LLMReply: ...


class LLMClient:

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 256, json_schema: Optional[Dict[str, Any]] = None,
             purpose: str = "chat") -> LLMReply:
        body: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        if json_schema is not None:
            # llama-server constrains the output to this JSON schema, so the reply always parses
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "item", "schema": json_schema}}
        started = time.perf_counter()
        try:
            response = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as error:
            log_event("llm_call", status="error", error_type=type(error).__name__, stage=purpose)
            raise LLMError(f"model server error: {type(error).__name__}") from error
        usage = data.get("usage", {})
        reply = LLMReply(text=text.strip(), tokens_in=usage.get("prompt_tokens", 0), tokens_out=usage.get("completion_tokens", 0),
                         duration_ms=round((time.perf_counter() - started) * 1000))
        log_event("llm_call", status="ok", model=self.model, tokens_in=reply.tokens_in, tokens_out=reply.tokens_out,
                  duration_ms=reply.duration_ms, stage=purpose)
        return reply
