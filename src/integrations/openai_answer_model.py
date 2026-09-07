from __future__ import annotations

import os
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part)


def _message_role(message: BaseMessage) -> str:
    if message.type == "ai":
        return "assistant"
    return "user"


def _prompt_payload(value: Any) -> tuple[Optional[str], list[dict[str, str]]]:
    if isinstance(value, PromptValue):
        messages = value.to_messages()
    elif isinstance(value, list) and all(isinstance(item, BaseMessage) for item in value):
        messages = value
    elif isinstance(value, BaseMessage):
        messages = [value]
    else:
        return None, [{"role": "user", "content": str(value)}]

    instructions: list[str] = []
    input_messages: list[dict[str, str]] = []
    for message in messages:
        text = _content_text(message.content).strip()
        if not text:
            continue
        if message.type == "system":
            instructions.append(text)
        else:
            input_messages.append(
                {"role": _message_role(message), "content": text}
            )
    return "\n\n".join(instructions) or None, input_messages


class OpenAIResponsesAnswerModel(Runnable[Any, AIMessage]):
    """Minimal LangChain-compatible adapter for answer-only OpenAI calls."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: float,
        max_retries: int,
        json_schema: Optional[dict[str, Any]] = None,
        client_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured for online answer generation."
            )

        if client_factory is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - deployment dependency
                raise RuntimeError(
                    "The 'openai' package is required for online answer generation."
                ) from exc
            client_factory = OpenAI

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout_seconds,
            "max_retries": max(0, max_retries),
        }
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        if base_url:
            client_kwargs["base_url"] = base_url

        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.json_schema = json_schema
        self.client = client_factory(**client_kwargs)

    def invoke(
        self,
        input: Any,
        config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        del config, kwargs
        instructions, input_messages = _prompt_payload(input)
        request: dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            # CRM and insurance content must not be retained for model training.
            "store": False,
        }
        if instructions:
            request["instructions"] = instructions
        if self.json_schema is not None:
            request["text"] = {"format": {
                "type": "json_schema", "name": "groundedness", "strict": True,
                "schema": self.json_schema,
            }}

        response = self.client.responses.create(**request)
        content = str(getattr(response, "output_text", "") or "")
        metadata: dict[str, Any] = {
            "provider": "openai",
            "model": self.model,
        }
        response_id = getattr(response, "id", None)
        if response_id:
            metadata["response_id"] = response_id

        return AIMessage(content=content, response_metadata=metadata)
