from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from scripts.experimental_groundedness_v5 import (
    EXPERIMENTAL_GROUNDING_ALGORITHM_VERSION,
    calculate_groundedness_score_v5_experimental,
)
from src.guardrails.integrations.nemo_actions import configure_grounding_embeddings


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


def test_v5_can_verify_a_german_claim_against_english_evidence() -> None:
    class CrossLingualEmbeddingStub:
        def embed_documents(self, texts):
            vectors = []
            for text in texts:
                normalized = text.casefold()
                if any(
                    marker in normalized
                    for marker in (
                        "vertrag läuft ein jahr",
                        "contract lasts one year",
                        "automatisch um ein weiteres jahr",
                        "automatically renews for another year",
                    )
                ):
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            return vectors

    docs = [
        Document(
            page_content=(
                "The contract lasts one year and automatically renews for another "
                "year unless it is terminated on time."
            ),
            metadata={"source_file": "rental-policy.pdf", "source_page": 3},
        )
    ]
    configure_grounding_embeddings(CrossLingualEmbeddingStub())
    try:
        result = calculate_groundedness_score_v5_experimental(
            (
                "Der Vertrag läuft ein Jahr und verlängert sich automatisch um ein "
                "weiteres Jahr [source: rental-policy.pdf, page 3]."
            ),
            docs,
            "Wie lange läuft der Vertrag und wie verlängert er sich?",
        )
    finally:
        configure_grounding_embeddings(None)

    assert result.score > 0.7888
    assert not result.citation_mismatches
    assert not result.unsupported_atomic_facts


def test_v5_still_caps_wrong_cross_lingual_duration() -> None:
    class TopicOnlyEmbeddingStub:
        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

    docs = [
        Document(
            page_content="The contract lasts one year.",
            metadata={"source_file": "rental-policy.pdf", "source_page": 3},
        )
    ]
    configure_grounding_embeddings(TopicOnlyEmbeddingStub())
    try:
        result = calculate_groundedness_score_v5_experimental(
            "Der Vertrag läuft zwei Jahre [rental-policy.pdf, page 3].",
            docs,
            "Wie lange läuft der Vertrag?",
        )
    finally:
        configure_grounding_embeddings(None)

    assert result.score <= 0.38
    assert "duration_number:2" in result.unsupported_atomic_facts


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


def test_v5_ignores_multiple_citations_on_document_source_line() -> None:
    docs = [
        Document(
            page_content="Water damage to household contents is covered.",
            metadata={"source_file": "household.pdf", "source_page": 9},
        )
    ]
    answer = (
        "Water damage to household contents is covered.\n"
        "Document sources: [household.pdf, page 10] [household.pdf, page 10]"
    )

    result = calculate_groundedness_score_v5_experimental(
        answer,
        docs,
        "Is water damage covered?",
    )

    assert "number:10" not in result.unsupported_atomic_facts
    assert result.extracted_claims == ("Water damage to household contents is covered.",)


def test_v5_normalizes_english_calendar_date_to_iso_context_date() -> None:
    docs = [
        Document(
            page_content="The policy ended on 2025-12-31.",
            metadata={"source_file": "crm.txt"},
        )
    ]

    result = calculate_groundedness_score_v5_experimental(
        "The policy ended on December 31, 2025.",
        docs,
        "When did the policy end?",
    )

    assert "date:2025-12-31" not in result.unsupported_atomic_facts
    assert not {
        fact for fact in result.unsupported_atomic_facts if fact.startswith("number:")
    }


def test_v5_matches_compact_duration_units_to_written_units() -> None:
    docs = [
        Document(
            page_content="The service includes 24/7 assistance and 48h claims payment.",
            metadata={"source_file": "services.pdf"},
        )
    ]

    result = calculate_groundedness_score_v5_experimental(
        "The service provides 24/7 assistance and claims payment within 48 hours.",
        docs,
        "Which services are provided?",
    )

    assert "number:48" not in result.unsupported_atomic_facts


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


def test_v5_ignores_multi_page_source_line_and_policy_advice() -> None:
    docs = [
        Document(
            page_content="A three month waiting period applies to the legal dispute.",
            metadata={"source_file": "legal.pdf", "source_page": 2},
        )
    ]
    answer = """A three month waiting period applies to the legal dispute.
For specific conditions, please refer to your policy details.
Source: [legal.pdf, page 3; page 4; page 5]"""

    result = calculate_groundedness_score_v5_experimental(
        answer,
        docs,
        "Which waiting period applies?",
    )

    assert result.extracted_claims == (
        "A three month waiting period applies to the legal dispute.",
    )
    assert {
        item["reason_code"] for item in result.ignored_nonsemantic_lines
    } >= {"advisory_boilerplate", "source_heading"}


def test_v5_matches_natural_date_and_compact_time_unit() -> None:
    docs = [
        Document(
            page_content=(
                "Policy TEST-PHV-2025-1301 ended on 2025-12-31. "
                "Claims payment is made within 48h following successful review."
            ),
            metadata={},
        )
    ]

    result = calculate_groundedness_score_v5_experimental(
        (
            "Policy TEST-PHV-2025-1301 ended on December 31, 2025. "
            "Claims payment is made within 48 hours following successful review."
        ),
        docs,
        "When did the policy end and when is payment made?",
    )

    assert not result.unsupported_atomic_facts
    assert result.score >= 0.7888


def test_v5_recovers_positive_scope_from_multicolumn_liability_table() -> None:
    docs = [
        Document(
            page_content=(
                "Personal liability insurance. Statutory liability is insured and "
                "defence against unjustified claims. Property damage. Purely financial "
                "losses. The following claims are not insured. Q1 Liability claims "
                "for damages from third parties on the basis of statutory liability "
                "due to destruction, damage or loss of property. Q2 costs for defending "
                "against unjustified claims. A36 contractual liability beyond the "
                "scope of statutory liability."
            ),
            metadata={},
        )
    ]

    result = calculate_groundedness_score_v5_experimental(
        (
            "Personal liability insurance covers statutory third-party liability "
            "and property damage, and defence against unjustified claims."
        ),
        docs,
        "Is third-party property damage generally covered?",
    )

    assert not result.polarity_mismatches
    assert result.score >= 0.7888


def test_v5_ignores_inline_numbered_headings_as_presentation_markers() -> None:
    docs = [
        Document(
            page_content=(
                "Contract formation and duration are regulated. Duties and benefits "
                "in the event of a claim are regulated."
            ),
            metadata={},
        )
    ]
    answer = """1. Contract formation and duration are regulated.
2. Duties and benefits in the event of a claim are regulated."""

    result = calculate_groundedness_score_v5_experimental(answer, docs, "What is regulated?")

    assert "number:1" not in result.unsupported_atomic_facts
    assert "number:2" not in result.unsupported_atomic_facts
    assert result.score >= 0.7888


def test_v5_reconstructs_legal_exclusion_heading_across_table_rows() -> None:
    docs = [
        Document(
            page_content=(
                "The insurance does not cover A8 relating to any legal protection "
                "claims and characteristics not specifically named; A9 relating to "
                "cases arising before conclusion of the insurance contract or during "
                "the waiting period."
            ),
            metadata={},
        )
    ]
    answer = (
        "The insurance does not cover legal-protection claims not specifically named; "
        "further subject-specific exclusions apply. The insurance does not cover "
        "legal-protection cases arising before conclusion of the insurance contract "
        "or during the waiting period."
    )

    result = calculate_groundedness_score_v5_experimental(
        answer,
        docs,
        "Which waiting-period exclusions apply?",
    )

    assert not result.polarity_mismatches
    assert result.score >= 0.7888


def test_v5_reconstructs_household_water_scope_across_ocr_break() -> None:
    docs = [
        Document(
            page_content=(
                "Household contents. Destruction, damage or loss as a result of E1 "
                "leakage of liq uids and gas: a) from pipelines or connected "
                "installations or apparatus."
            ),
            metadata={},
        )
    ]
    answer = (
        "Household water cover includes leakage of liquids and gas from pipelines, "
        "connected installations or apparatus. The water peril concerns destruction, "
        "damage or loss of insured household contents."
    )

    result = calculate_groundedness_score_v5_experimental(
        answer,
        docs,
        "Does household contents insurance cover water damage?",
    )

    assert result.score >= 0.7888
