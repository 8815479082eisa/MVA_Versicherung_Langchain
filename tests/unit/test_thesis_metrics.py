from __future__ import annotations

from src.evaluation.security_eval import derive_case_label, summarize_security_rows
from src.evaluation.thesis_metrics import build_qa_metric_row, lexical_support_score


def test_lexical_support_score_is_high_for_supported_claim() -> None:
    claim = "Health insurance helps pay for medical costs."
    passage = "Health insurance helps pay for medical costs and treatments."

    score = lexical_support_score(claim, passage)

    assert score >= 0.8


def test_build_qa_metric_row_computes_retrieval_answer_and_citation_metrics() -> None:
    row = build_qa_metric_row(
        question="What is health insurance?",
        references=["Health insurance helps pay for medical costs."],
        prediction="Health insurance helps pay for medical costs.",
        context_docs=[
            {"page_content": "Health insurance helps pay for medical costs and treatments."},
            {"page_content": "Retirement plans build future savings."},
        ],
        sources=[{"document_id": "doc-1"}],
        audit_row={"retrieval_needed": "RETRIEVE", "safety_decision": "allow", "latency_ms": 12},
        support_threshold=0.2,
    )

    assert row["metrics"]["retrieval_support_hit"] == 1.0
    assert row["metrics"]["retrieval_context_precision"] == 0.5
    assert row["metrics"]["answer_exact_match"] == 1.0
    assert row["metrics"]["answer_token_f1"] == 1.0
    assert row["metrics"]["citation_source_presence"] == 1.0
    assert row["metrics"]["citation_support_rate"] == 1.0
    assert row["diagnostics"]["inline_citation_coverage"] == 0.0


def test_build_qa_metric_row_handles_missing_support() -> None:
    row = build_qa_metric_row(
        question="What is liability insurance?",
        references=["Liability insurance covers damages to third parties."],
        prediction="It is something about retirement savings.",
        context_docs=[{"page_content": "Retirement plans build future savings."}],
        sources=[],
        audit_row={"retrieval_needed": "RETRIEVE", "safety_decision": "allow"},
        support_threshold=0.2,
    )

    assert row["metrics"]["retrieval_support_hit"] == 0.0
    assert row["metrics"]["retrieval_context_precision"] == 0.0
    assert row["metrics"]["citation_source_presence"] == 0.0
    assert row["metrics"]["citation_support_rate"] == 0.0


def test_build_qa_metric_row_marks_grounding_fallback_as_fallback_answer() -> None:
    row = build_qa_metric_row(
        question="What is liability insurance?",
        references=["Liability insurance covers damages to third parties."],
        prediction="I could not generate an answer that is sufficiently supported by the available documents.",
        context_docs=[{"page_content": "Liability insurance covers damages to third parties."}],
        sources=[],
        audit_row={"retrieval_needed": "RETRIEVE", "safety_decision": "post_fallback"},
        support_threshold=0.2,
    )

    assert row["diagnostics"]["fallback_answer"] is True


def test_security_helpers_derive_labels_and_summary() -> None:
    assert derive_case_label({"expected_decision": "flag_or_block"}) == "attack"
    assert derive_case_label({"expected_decision": "allow"}) == "benign"
    assert derive_case_label({"expected_decision": "review_or_allow"}) == "benign"

    summary = summarize_security_rows(
        [
            {"metrics": {"attack_block": 1.0, "benign_allow": None}},
            {"metrics": {"attack_block": 0.0, "benign_allow": None}},
            {"metrics": {"attack_block": None, "benign_allow": 1.0}},
            {"metrics": {"attack_block": None, "benign_allow": 0.0}},
        ]
    )

    assert summary["attack_block_rate"] == 0.5
    assert summary["benign_allow_rate"] == 0.5
    assert summary["false_negative_count"] == 1
    assert summary["false_positive_count"] == 1
