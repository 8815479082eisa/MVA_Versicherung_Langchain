from __future__ import annotations

import json

from src.api import rag_service
from src.utils.language import detect_response_language


def test_detects_language_from_final_answer() -> None:
    assert detect_response_language("The windshield damage is covered under this contract.") == "English"
    assert detect_response_language("Der Glasschaden ist durch die Versicherung gedeckt.") == "German"


def test_short_empty_or_marker_free_answer_uses_documented_fallback() -> None:
    assert detect_response_language("") == "Unknown"
    assert detect_response_language("[policy:1]") == "Unknown"
    assert detect_response_language("150 EUR") == "Unknown"
    assert detect_response_language("150 EUR", fallback="English") == "English"


def test_audit_language_comes_from_answer_not_query_or_extra_field(tmp_path, monkeypatch) -> None:
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(rag_service, "AUDIT_LOG_FILE", str(audit_path))

    rag_service.audit_log(
        query="Ist der Schaden gedeckt?",
        retrieved_documents=[],
        compressed_context=[],
        generated_answer="The windshield damage is covered under partial comprehensive insurance.",
        response_language="German",
    )
    rag_service.audit_log(
        query="Is the damage covered?",
        retrieved_documents=[],
        compressed_context=[],
        generated_answer="Der Glasschaden ist durch die Versicherung gedeckt.",
        response_language="English",
    )

    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert [record["response_language"] for record in records] == ["English", "German"]
