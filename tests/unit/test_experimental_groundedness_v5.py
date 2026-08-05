from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from scripts.experimental_groundedness_v5 import (
    EXPERIMENTAL_GROUNDING_ALGORITHM_VERSION,
    calculate_groundedness_score_v5_experimental,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASES = PROJECT_ROOT / "tests" / "fixtures" / "groundedness_extended_candidates.jsonl"


def _case(case_id: str) -> dict:
    for line in CASES.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["case_id"] == case_id:
            return row
    raise KeyError(case_id)


def _score(case_id: str):
    case = _case(case_id)
    docs = [
        Document(
            page_content=text,
            metadata={"source_file": case["source_file"], "source_page": case["source_page"]},
        )
        for text in case["context"]
    ]
    return calculate_groundedness_score_v5_experimental(
        case["candidate_answer"], docs, case["question"]
    )


def test_v5_version_is_explicitly_experimental() -> None:
    assert EXPERIMENTAL_GROUNDING_ALGORITHM_VERSION == "fact_aware_claim_support_v5_experimental"


def test_v5_caps_single_wrong_percentage_in_long_answer() -> None:
    result = _score("ext-iqa-fail-071")
    assert result.base_v4_score > 0.9
    assert result.score <= 0.38
    assert "percent:60" in result.unsupported_atomic_facts


def test_v5_detects_wrong_pdf_limit() -> None:
    result = _score("ext-helvetia-fail-005")
    assert result.base_v4_score == 1.0
    assert result.score <= 0.38
    assert result.unsupported_atomic_facts


def test_v5_detects_citation_page_mismatch() -> None:
    result = _score("ext-helvetia-fail-024")
    assert result.score <= 0.35
    assert "citation_page:3" in result.citation_mismatches


def test_v5_does_not_treat_valid_citation_page_as_unsupported_number() -> None:
    result = _score("ext-helvetia-pass-045")
    assert "number:4" not in result.unsupported_atomic_facts
    assert not result.citation_mismatches


def test_v5_selects_lara_neumanns_current_motor_policy() -> None:
    supported = _score("ext-crm-pass-049")
    wrong_current_policy = _score("ext-crm-fail-091")
    assert supported.score > 0.78
    assert not supported.structured_mismatches
    assert wrong_current_policy.score <= 0.30
    assert any(item.startswith("current_policy:") for item in wrong_current_policy.structured_mismatches)


def test_v5_does_not_parse_helvetia_customer_service_as_customer() -> None:
    docs = [
        Document(
            page_content=(
                "Customer: Lara Neumann. Policy number: TEST-KFZ-2026-1003. "
                "Coverage type: Partial Coverage. Status: Active."
            ),
            metadata={},
        ),
        Document(
            page_content=(
                "These additional benefits are only provided if Helvetia Customer "
                "Service was notified and services were arranged through a Helvetia partner."
            ),
            metadata={},
        ),
    ]
    result = calculate_groundedness_score_v5_experimental(
        "Helvetia Customer Service must be notified. Lara Neumann's policy has Partial Coverage.",
        docs,
        "What conditions apply to Lara Neumann's current policy?",
    )

    assert "structured_customer:service must" not in result.structured_mismatches


def test_v5_caps_conflicting_coverage_taxonomy_terms() -> None:
    docs = [
        Document(
            page_content=(
                "Customer: Lara Neumann. Policy number: TEST-KFZ-2026-1003. "
                "Coverage type: Partial Coverage. Status: Active."
            ),
            metadata={},
        )
    ]
    result = calculate_groundedness_score_v5_experimental(
        "Lara Neumann's policy has Comprehensive Coverage.",
        docs,
        "What is Lara Neumann's policy coverage type?",
    )

    assert result.score <= 0.30
    assert any(
        item.startswith("coverage_taxonomy:part_comprehensive!=full_comprehensive")
        for item in result.structured_mismatches
    )


def test_v5_caps_ambiguous_comprehensive_for_partial_coverage_policy() -> None:
    docs = [
        Document(
            page_content=(
                "Customer: Lara Neumann. Policy number: TEST-KFZ-2026-1003. "
                "Coverage type: Partial Coverage. Status: Active."
            ),
            metadata={},
        )
    ]
    result = calculate_groundedness_score_v5_experimental(
        "Lara Neumann's policy is covered under comprehensive insurance.",
        docs,
        "What is Lara Neumann's policy coverage type?",
    )

    assert result.score <= 0.30
    assert any(
        item.startswith("coverage_taxonomy:part_comprehensive!=ambiguous_comprehensive")
        for item in result.structured_mismatches
    )


def test_v5_allows_part_comprehensive_term_for_partial_coverage_policy() -> None:
    docs = [
        Document(
            page_content=(
                "Customer: Lara Neumann. Policy number: TEST-KFZ-2026-1003. "
                "Coverage type: Partial Coverage. Status: Active."
            ),
            metadata={},
        )
    ]
    result = calculate_groundedness_score_v5_experimental(
        "Lara Neumann's policy has part comprehensive insurance.",
        docs,
        "What is Lara Neumann's policy coverage type?",
    )

    assert not result.structured_mismatches
    assert "structured_field_mismatch_cap_0.30" not in result.applied_caps


def test_v5_ignores_markdown_source_labels_citations_and_decision_disclaimer() -> None:
    docs = [
        Document(
            page_content=(
                "Policy number: TEST-KFZ-2026-1003. Coverage type: Partial Coverage. "
                "Individual deductible: 300 EUR. Annual premium: 720 EUR."
            ),
            metadata={"source_file": "crm.txt"},
        ),
        Document(
            page_content=(
                "Glass damage to a front windscreen is covered. You will not have "
                "to bear a deductible if the damaged front windscreen is repaired "
                "and not replaced."
            ),
            metadata={"source_file": "motor.pdf", "source_page": 9},
        ),
    ]
    answer = """### Individual contract data
- Policy number: TEST-KFZ-2026-1003
- Coverage type: Partial Coverage
- Individual deductible: 300 EUR
- Annual premium: 720 EUR

### General insurance terms
Glass damage to the front windscreen is covered [motor.pdf, page 9].
No deductible is charged if the damaged front windscreen is repaired and not replaced [motor.pdf, page 9].

**Sources:**
- [motor.pdf, page 9]
- [CRM: TEST-KFZ-2026-1003]

This general information does not constitute a final claim decision."""

    result = calculate_groundedness_score_v5_experimental(
        answer,
        docs,
        "What does the current motor policy cover?",
    )

    assert result.minimum_claim_support > 0.7
    assert result.score >= 0.7888
    assert not result.citation_mismatches


def test_v5_ignores_source_line_with_trailing_period_after_citation() -> None:
    docs = [
        Document(
            page_content="Theft is listed as an insured benefit with partially comprehensive insurance.",
            metadata={"source_file": "motor.pdf", "source_page": 3},
        )
    ]
    answer = "- Source: [motor.pdf, page 3]."

    result = calculate_groundedness_score_v5_experimental(
        answer,
        docs,
        "Is theft covered?",
    )

    assert not result.extracted_claims
    assert any(
        item["reason_code"] == "source_heading"
        for item in result.ignored_nonsemantic_lines
    )


def test_v5_rejects_wrong_deductible_polarity_despite_clean_formatting() -> None:
    docs = [
        Document(
            page_content=(
                "You will not have to bear a deductible if the damaged front "
                "windscreen is repaired and not replaced."
            ),
            metadata={"source_file": "motor.pdf", "source_page": 9},
        )
    ]

    result = calculate_groundedness_score_v5_experimental(
        "The deductible still applies when the windscreen is repaired rather than replaced.",
        docs,
        "Does the deductible apply to a windscreen repair?",
    )

    assert result.score < 0.7888


def test_v5_accepts_windows_metadata_paths_on_linux_style_runtime() -> None:
    docs = [
        Document(
            page_content="Glass damage is covered.",
            metadata={
                "source": r"data\raw\pdfs\motor-vehicle-insurance-sti.pdf",
                "source_page": 14,
            },
        )
    ]

    result = calculate_groundedness_score_v5_experimental(
        "Glass damage is covered [motor-vehicle-insurance-sti.pdf, page 14].",
        docs,
        "Is glass damage covered?",
    )

    assert not result.citation_mismatches
    assert "citation_mismatch_cap_0.35" not in result.applied_caps


def test_v5_ignores_information_is_general_decision_disclaimer() -> None:
    docs = [
        Document(
            page_content="Glass damage is covered.",
            metadata={"source_file": "motor.pdf", "source_page": 0},
        )
    ]
    disclaimer = "This information is general and should not be considered a final claim decision."

    result = calculate_groundedness_score_v5_experimental(
        f"Glass damage is covered.\n\n{disclaimer}",
        docs,
        "Is glass damage covered?",
    )

    assert result.extracted_claims == ("Glass damage is covered.",)
    assert {
        item["reason_code"] for item in result.ignored_nonsemantic_lines
    } >= {"decision_disclaimer"}

    alternate_disclaimer = (
        "This information provides a general overview without making a final claim decision."
    )
    alternate = calculate_groundedness_score_v5_experimental(
        f"Glass damage is covered.\n\n{alternate_disclaimer}",
        docs,
        "Is glass damage covered?",
    )
    assert alternate.extracted_claims == ("Glass damage is covered.",)
    assert {
        item["reason_code"] for item in alternate.ignored_nonsemantic_lines
    } >= {"decision_disclaimer"}

    combined_decision_disclaimer = (
        "This is general information and not a final claim or coverage decision."
    )
    combined = calculate_groundedness_score_v5_experimental(
        f"Glass damage is covered.\n\n{combined_decision_disclaimer}",
        docs,
        "Is glass damage covered?",
    )
    assert combined.extracted_claims == ("Glass damage is covered.",)
    assert combined.minimum_claim_support == 1.0


def test_v5_ignores_short_bold_section_labels() -> None:
    docs = [
        Document(
            page_content=(
                "Policy number: TEST-KFZ-2026-1003. Glass damage is covered."
            ),
            metadata={"source_file": "motor.pdf"},
        )
    ]
    answer = """**CRM Facts:**
- Policy number: TEST-KFZ-2026-1003

**General Policy Conditions:**
Glass damage is covered.

**Conditions:**"""

    result = calculate_groundedness_score_v5_experimental(
        answer,
        docs,
        "Is glass damage covered under the current policy?",
    )

    assert "CRM Facts:" not in result.extracted_claims
    assert "General Policy Conditions:" not in result.extracted_claims
    assert "Conditions:" not in result.extracted_claims
    assert sum(
        item["reason_code"] == "section_heading"
        for item in result.ignored_nonsemantic_lines
    ) == 3


def test_v5_accepts_supported_claim_spanning_adjacent_chunks() -> None:
    docs = [
        Document(
            page_content="G10.2 You will not have to bear a deductible:",
            metadata={"source_file": "motor.pdf", "source_page": 8},
        ),
        Document(
            page_content=(
                "g) if the damaged front windscreen is repaired and not replaced "
                "in the case of glass damage."
            ),
            metadata={"source_file": "motor.pdf", "source_page": 8},
        ),
    ]

    result = calculate_groundedness_score_v5_experimental(
        (
            "You will not have to bear a deductible if the damaged front windscreen "
            "is repaired and not replaced in the case of glass damage."
        ),
        docs,
        "Is the deductible charged when the windscreen is repaired?",
    )

    assert result.base_v4_score < 0.7888
    assert result.average_claim_support == 1.0
    assert result.score >= 0.7888


def test_v5_allows_supported_coverage_with_supported_noncoverage_condition() -> None:
    docs = [
        Document(
            page_content=(
                "Glass damage to the front windscreen is covered when repair is "
                "necessary for safety reasons."
            ),
            metadata={},
        ),
        Document(
            page_content=(
                "No compensation will be paid if the replacement or repair is not "
                "carried out."
            ),
            metadata={},
        ),
    ]

    result = calculate_groundedness_score_v5_experimental(
        (
            "Glass damage to the front windscreen is covered when repair is necessary "
            "for safety reasons. No compensation will be paid if the repair is not "
            "carried out."
        ),
        docs,
        "Is windscreen damage covered?",
    )

    assert not result.polarity_mismatches


def test_v5_ignores_numbered_list_markers_and_review_disclaimer() -> None:
    docs = [
        Document(
            page_content="Glass damage is covered.",
            metadata={},
        )
    ]
    answer = """**Conditions:**
1.
Glass damage is covered.
2.
For a definitive decision on any claim, further review is required."""

    result = calculate_groundedness_score_v5_experimental(
        answer,
        docs,
        "Is glass damage covered?",
    )

    assert result.extracted_claims == ("Glass damage is covered.",)
    assert "number:1" not in result.unsupported_atomic_facts
    assert "number:2" not in result.unsupported_atomic_facts
    assert {
        item["reason_code"] for item in result.ignored_nonsemantic_lines
    } >= {"section_heading", "list_enumerator", "decision_disclaimer"}
