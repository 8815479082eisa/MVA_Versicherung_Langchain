from __future__ import annotations

from pathlib import Path

from src.evaluation.reference_answer_quality import (
    CRMReferenceStore,
    evaluate_case,
    evaluate_concepts,
    load_jsonl,
    load_reference_spec,
    summarize_evaluations,
    validate_spec_coverage,
    value_present,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "benchmarks" / "routing" / "routing_eval_80.jsonl"
SPEC = ROOT / "data" / "benchmarks" / "answer_quality" / "reference_spec_v1.json"
CRM_DIR = ROOT / "data" / "synthetic" / "crm"


def _case(case_id: str):
    return next(case for case in load_jsonl(DATASET) if case["id"] == case_id)


def _response(*, route: str, answer: str, sources=None, status_code: int = 200):
    return {
        "status_code": status_code,
        "latency_ms": 12.5,
        "payload": {
            "route": route,
            "status": "complete",
            "answer": answer,
            "sources": sources or [],
            "diagnostics": {
                "evidence": {
                    "groundedness": {"score": 0.95, "threshold": 0.7888, "passed": True}
                }
            },
        },
        "collection_error": None,
    }


def _evaluate(case_id: str, response):
    spec = load_reference_spec(SPEC)
    case = _case(case_id)
    return evaluate_case(
        case,
        response,
        spec["cases"][case_id],
        spec,
        CRMReferenceStore(CRM_DIR),
    )


def test_reference_spec_covers_all_80_cases():
    cases = load_jsonl(DATASET)
    spec = load_reference_spec(SPEC)
    assert len(cases) == 80
    assert validate_spec_coverage(cases, spec) == []


def test_reference_spec_allows_an_explicit_rerun_subset():
    cases = [_case("route-crm-010"), _case("route-denied-002")]
    spec = load_reference_spec(SPEC)

    assert validate_spec_coverage(cases, spec, allow_extra_specs=True) == []


def test_value_present_accepts_common_date_and_numeric_formats():
    assert value_present("The recorded premium is 684 EUR.", "684")
    assert value_present("The policy ended on 31.12.2025.", "2025-12-31")
    assert not value_present("The premium is 685 EUR.", "684")


def test_concept_matching_handles_inflection_and_function_words():
    catalog = {
        "fire": {
            "description": "Fire cover",
            "alternatives": ["household contents are covered against fire"],
        }
    }
    rows = evaluate_concepts(
        "The household policy covers fire damage to its insured contents.",
        ["fire"],
        catalog,
        threshold=0.75,
    )

    assert rows[0]["covered"] is True


def test_concept_matching_normalizes_plural_costs_lawyers_and_regulations():
    catalog = {
        "legal": {
            "description": "Legal costs",
            "alternatives": ["lawyer or legal assistance costs"],
        },
        "contract": {
            "description": "Contract rules",
            "alternatives": ["premium and termination rules"],
        },
    }
    rows = evaluate_concepts(
        (
            "Legal-protection benefits pay the cost of lawyers engaged. "
            "The regulations cover premiums and termination."
        ),
        ["legal", "contract"],
        catalog,
        threshold=0.75,
    )

    assert [row["covered"] for row in rows] == [True, True]


def test_crm_only_uses_synthetic_csv_ground_truth():
    result = _evaluate(
        "route-crm-001",
        _response(
            route="crm-only",
            answer="CRM facts: The annual premium is 684 EUR.",
            sources=[
                {
                    "documentId": "espocrm:MvaPolicy:TEST-KFZ-2026-1001",
                    "documentTitle": "EspoCRM policy",
                }
            ],
        ),
    )
    assert result["crm_fact_recall"] == 1.0
    assert result["overall_pass"] is True


def test_rag_only_checks_requirements_source_and_groundedness():
    result = _evaluate(
        "route-rag-001",
        _response(
            route="retrieval-only",
            answer=(
                "Partially comprehensive glass insurance covers involuntary glass breakage "
                "to the front and rear windscreens, side windows and sunroof when repair or "
                "replacement is required. Source: [motor-vehicle-insurance-sti.pdf, page 14]"
            ),
            sources=[
                {
                    "documentId": "motor-vehicle-insurance-sti.pdf",
                    "documentTitle": "motor-vehicle-insurance-sti.pdf",
                    "page": 14,
                    "snippet": (
                        "The glass insurance covers involuntary breakage and accident damage "
                        "to front and rear windscreens, side windows and the sunroof."
                    ),
                }
            ],
        ),
    )
    assert result["requirement_recall"] == 1.0
    assert result["expected_document_source_present"] is True
    assert result["groundedness_pass"] is True
    assert result["overall_pass"] is True


def test_combined_requires_both_crm_and_document_evidence():
    result = _evaluate(
        "route-combined-001",
        _response(
            route="combined",
            answer=(
                "CRM facts: The recorded deductible is 150 EUR. Document evidence: Glass "
                "insurance generally covers involuntary glass breakage to windscreens, side "
                "windows and the sunroof when repair or replacement is required. "
                "Source: [motor-vehicle-insurance-sti.pdf, page 14]"
            ),
            sources=[
                {
                    "documentId": "espocrm:MvaPolicy:TEST-KFZ-2026-1001",
                    "documentTitle": "EspoCRM policy",
                    "snippet": "deductible 150 EUR",
                },
                {
                    "documentId": "motor-vehicle-insurance-sti.pdf",
                    "documentTitle": "motor-vehicle-insurance-sti.pdf",
                    "page": 14,
                    "snippet": "Glass insurance covers involuntary breakage to windscreens and side windows.",
                },
            ],
        ),
    )
    assert result["crm_fact_recall"] == 1.0
    assert result["requirement_recall"] == 1.0
    assert result["overall_pass"] is True


def test_extra_retrieval_result_is_not_treated_as_a_wrong_citation():
    result = _evaluate(
        "route-rag-001",
        _response(
            route="retrieval-only",
            answer=(
                "Glass insurance covers involuntary glass breakage to front and rear "
                "windscreens, side windows and the sunroof when repair is required. "
                "Source: [motor-vehicle-insurance-sti.pdf, page 14]"
            ),
            sources=[
                {
                    "documentTitle": "motor-vehicle-insurance-product-sheet.pdf",
                    "snippet": "General motor product information.",
                },
                {
                    "documentTitle": "motor-vehicle-insurance-sti.pdf",
                    "snippet": "Glass insurance covers involuntary windscreen breakage.",
                },
            ],
        ),
    )
    assert result["retrieval_expected_file_precision"] == 0.5
    assert result["citation_expected_file_precision"] == 1.0
    assert result["citation_returned_source_link_precision"] == 1.0
    assert result["overall_pass"] is True


def test_wrong_explicit_document_citation_fails_citation_precision():
    result = _evaluate(
        "route-rag-001",
        _response(
            route="retrieval-only",
            answer=(
                "Glass insurance covers involuntary glass breakage to front and rear "
                "windscreens, side windows and the sunroof. "
                "Source: [buildings-insurance-sti.pdf, page 5]"
            ),
            sources=[
                {
                    "documentTitle": "motor-vehicle-insurance-sti.pdf",
                    "snippet": "Glass insurance covers involuntary windscreen breakage.",
                }
            ],
        ),
    )
    assert result["citation_expected_document_present"] is False
    assert result["citation_expected_file_precision"] == 0.0
    assert result["citation_returned_source_link_precision"] == 0.0
    assert "citation_not_linked_to_returned_source" in result["errors"]
    assert result["overall_pass"] is False


def test_denied_case_fails_if_it_is_misrouted_and_accesses_sources():
    result = _evaluate(
        "route-denied-002",
        _response(
            route="retrieval-only",
            answer="Here is a list of policies.",
            sources=[{"documentId": "some.pdf", "documentTitle": "some.pdf"}],
        ),
    )
    assert result["denied_no_data_access"] is False
    assert result["overall_pass"] is False
    assert "route_mismatch" in result["errors"]
    assert "denied_case_accessed_data" in result["errors"]


def test_summary_labels_results_as_not_human_validated():
    good = _evaluate(
        "route-crm-001",
        _response(route="crm-only", answer="The premium is 684 EUR."),
    )
    summary = summarize_evaluations([good])
    assert summary["human_validated"] is False
    assert summary["method_label"] == "automated_reference_based_technical_validation"
