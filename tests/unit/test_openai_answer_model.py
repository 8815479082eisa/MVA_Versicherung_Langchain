from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage

from src.integrations.openai_answer_model import OpenAIResponsesAnswerModel


class _FakeResponses:
    def __init__(self) -> None:
        self.request: dict | None = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text="Grounded answer.", id="resp_test")


class _FakeClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


def test_responses_adapter_requests_strict_judge_schema(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    client = _FakeClient()
    schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    model = OpenAIResponsesAnswerModel(
        model="gpt-4o-mini", temperature=0, max_output_tokens=2000,
        timeout_seconds=10, max_retries=0, json_schema=schema,
        client_factory=lambda **kwargs: client,
    )
    model.invoke("Judge a claim")
    assert client.responses.request["text"]["format"] == {
        "type": "json_schema", "name": "groundedness", "strict": True, "schema": schema,
    }


def test_responses_adapter_maps_langchain_messages_without_storing_response(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    client = _FakeClient()
    client_options: dict = {}

    def client_factory(**kwargs):
        client_options.update(kwargs)
        return client

    model = OpenAIResponsesAnswerModel(
        model="gpt-4o-mini",
        temperature=0.2,
        max_output_tokens=384,
        timeout_seconds=30.0,
        max_retries=1,
        client_factory=client_factory,
    )
    response = model.invoke(
        [
            SystemMessage(content="Use only the supplied evidence."),
            HumanMessage(content="What is covered?"),
        ]
    )

    assert response.content == "Grounded answer."
    assert response.response_metadata["provider"] == "openai"
    assert client_options["api_key"] == "test-only-key"
    assert client.responses.request == {
        "model": "gpt-4o-mini",
        "input": [{"role": "user", "content": "What is covered?"}],
        "max_output_tokens": 384,
        "temperature": 0.2,
        "store": False,
        "instructions": "Use only the supplied evidence.",
    }


def test_responses_adapter_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        OpenAIResponsesAnswerModel(
            model="gpt-4o-mini",
            temperature=0.0,
            max_output_tokens=32,
            timeout_seconds=10.0,
            max_retries=0,
            client_factory=lambda **_: _FakeClient(),
        )
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("Missing API key must fail before an API request.")
