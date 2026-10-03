"""
Provider-agnostic LLM client with tool calling (Anthropic Claude or OpenAI).

    LLM_PROVIDER=anthropic|openai   LLM_API_KEY=...   LLM_MODEL=(optional)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from common.config import LLM_API_KEY, LLM_MODEL, LLM_PROVIDER

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-5.5"}


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


def llm_configured() -> bool:
    return bool(LLM_API_KEY) and LLM_PROVIDER in DEFAULT_MODELS


class AnthropicClient:
    def __init__(self, system: str, tools: list, api_key: str = LLM_API_KEY, model: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=90)
        self.model = model or LLM_MODEL or DEFAULT_MODELS["anthropic"]
        self.system = system
        self.tools = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        self.messages: list[dict] = []

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant_text(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def step(self) -> Turn:
        resp = self.client.messages.create(model=self.model, max_tokens=2000, system=self.system,
                                           tools=self.tools, messages=self.messages)
        self.messages.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in resp.content]})
        turn = Turn()
        for b in resp.content:
            if b.type == "text":
                turn.text += b.text
            elif b.type == "tool_use":
                turn.tool_calls.append(ToolCall(b.id, b.name, dict(b.input or {})))
        return turn

    def add_tool_results(self, results: list[tuple[ToolCall, str, bool]]) -> None:
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": c.id, "content": out, "is_error": err} for c, out, err in results]})


class OpenAIClient:
    def __init__(self, system: str, tools: list, api_key: str = LLM_API_KEY, model: str | None = None):
        import openai
        self.client = openai.OpenAI(api_key=api_key, max_retries=2, timeout=90)
        self.model = model or LLM_MODEL or DEFAULT_MODELS["openai"]
        self.tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                         "parameters": t.parameters}} for t in tools]
        self.messages: list[dict] = [{"role": "system", "content": system}]

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant_text(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def step(self) -> Turn:
        resp = self.client.chat.completions.create(model=self.model, messages=self.messages, tools=self.tools)
        msg = resp.choices[0].message
        self.messages.append(msg.model_dump(exclude_none=True))
        turn = Turn(text=msg.content or "")
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            turn.tool_calls.append(ToolCall(tc.id, tc.function.name, args))
        return turn

    def add_tool_results(self, results: list[tuple[ToolCall, str, bool]]) -> None:
        for c, out, _ in results:
            self.messages.append({"role": "tool", "tool_call_id": c.id, "content": out})


def make_client(system: str, tools: list):
    if not llm_configured():
        raise LLMUnavailable("No LLM_API_KEY configured")
    return (AnthropicClient if LLM_PROVIDER == "anthropic" else OpenAIClient)(system, tools)
