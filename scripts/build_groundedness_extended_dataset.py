from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSURANCEQA_PATH = PROJECT_ROOT / "data" / "benchmarks" / "qa" / "insuranceqa" / "data_insuranceqa_1000.jsonl"
HELvetia_DIR = PROJECT_ROOT / "data" / "raw" / "pdfs"
EXCLUDED_PDF_DIR = PROJECT_ROOT / "data" / "raw" / "excluded_pdfs"
CRM_DIR = PROJECT_ROOT / "data" / "synthetic" / "crm"
LEGACY_PATH = PROJECT_ROOT / "tests" / "fixtures" / "groundedness_calibration_cases.json"

OUTPUT_JSONL = PROJECT_ROOT / "tests" / "fixtures" / "groundedness_extended_candidates.jsonl"
OUTPUT_REVIEW_CSV = PROJECT_ROOT / "reports" / "groundedness_dataset_review.csv"
OUTPUT_SOURCE_AUDIT = PROJECT_ROOT / "reports" / "groundedness_extended_dataset_source_audit.md"
OUTPUT_DICTIONARY = PROJECT_ROOT / "reports" / "groundedness_dataset_data_dictionary.md"
OUTPUT_QUALITY = PROJECT_ROOT / "reports" / "groundedness_dataset_quality_report.md"
OUTPUT_LEAKAGE = PROJECT_ROOT / "reports" / "groundedness_dataset_leakage_report.md"
OUTPUT_SPLIT = PROJECT_ROOT / "reports" / "groundedness_dataset_split_plan.json"
OUTPUT_GAP = PROJECT_ROOT / "reports" / "groundedness_annotation_gap_report.md"

GENERATOR_VERSION = "groundedness_extended_candidates_v1"
RANDOM_SEED = 20260801
NEAR_DUPLICATE_THRESHOLD = 0.94
MAX_CONTEXT_CHARS = 6000
REFERENCE_DATE = "2026-08-01"

REVIEW_LABELS = ["PASS", "FAIL", "AMBIGUOUS", "EXCLUDE"]
CRITICALITY_VALUES = ["CRITICAL", "NON_CRITICAL", "REVIEW_REQUIRED"]
ERROR_CATEGORIES = [
    "wrong_customer",
    "wrong_policy",
    "wrong_current_policy",
    "wrong_policy_status",
    "wrong_deductible",
    "wrong_premium",
    "wrong_coverage",
    "reversed_coverage",
    "wrong_exclusion",
    "wrong_limit",
    "wrong_percentage",
    "wrong_date",
    "unsupported_extra_claim",
    "partially_supported_answer",
    "answer_from_distractor_document",
    "insufficient_evidence",
    "citation_mismatch",
    "cross_product_confusion",
    "old_document_instead_of_current",
    "correct_fact_but_wrong_entity",
    "correct_number_but_wrong_policy",
    "unsupported_claim_decision",
    "other",
    "none",
]

REVIEW_FIELDS = [
    "reviewer_1_label",
    "reviewer_2_label",
    "adjudicated_label",
    "reviewer_1_error_category",
    "reviewer_2_error_category",
    "adjudicated_error_category",
    "reviewer_1_criticality",
    "reviewer_2_criticality",
    "adjudicated_criticality",
    "reviewer_1_notes",
    "reviewer_2_notes",
    "adjudication_notes",
    "reviewer_1_confidence",
    "reviewer_2_confidence",
    "adjudicated_confidence",
]

BASE_FIELDS = [
    "case_id",
    "source_type",
    "source_file",
    "source_document_id",
    "source_page",
    "source_chunk_id",
    "question",
    "context",
    "candidate_answer",
    "generator_expected_label",
    "mutation_type",
    "question_family_id",
    "document_family_id",
    "customer_group_id",
    "policy_group_id",
    "product_group",
    *REVIEW_FIELDS,
    "annotation_status",
    "groundedness_score",
    "split",
    "leakage_group_id",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").casefold()).strip()


def _clean_pdf_text(text: str) -> str:
    text = (text or "").replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip(" \t-•")
        for sentence in re.split(r"(?<=[.!?])\s+|\s+[•]\s+", text)
        if 55 <= len(sentence.strip()) <= 520
    ]


def _first_sentence(text: str) -> str:
    parts = _sentences(text)
    if parts:
        return parts[0]
    return (text or "").strip()[:400]


def _product_group(text: str) -> str:
    value = _normalize(text)
    rules = [
        ("motor", r"\b(car|auto|automobile|motor|vehicle|driver|windscreen|windshield)\b"),
        ("health", r"\b(health|medical|medicare|medigap|dental|hospital|disability)\b"),
        ("life", r"\b(life insurance|death benefit|annuity)\b"),
        ("household", r"\b(home|household|renters|property|building|contents)\b"),
        ("liability", r"\b(liability|third party)\b"),
        ("travel_assistance", r"\b(travel|trip|baggage|assistance|cancellation)\b"),
        ("legal_protection", r"\b(legal|lawyer|court|litigation)\b"),
        ("retirement", r"\b(retirement|pension|401k)\b"),
        ("rental_guarantee", r"\b(rental|rent|landlord|tenant|lease)\b"),
    ]
    for name, pattern in rules:
        if re.search(pattern, value):
            return name
    return "general_insurance"


def _new_case(**values: Any) -> dict[str, Any]:
    case = {field: "" for field in BASE_FIELDS}
    case.update(
        {
            "source_page": None,
            "source_chunk_id": "",
            "customer_group_id": "",
            "policy_group_id": "",
            "annotation_status": "PENDING",
            "groundedness_score": None,
            "split": "",
            "leakage_group_id": "",
            "base_leakage_group_id": "",
            "language": "English",
            "hard_negative": False,
            "mutation_source_file": "",
            "mutation_source_document_id": "",
            "mutation_source_page": None,
            "mutation_detail": "",
            "source_trace": "",
            "quality_flags": [],
            "excluded": False,
            "exclusion_reason": "",
            "generator_version": GENERATOR_VERSION,
            "generator_seed": RANDOM_SEED,
        }
    )
    case.update(values)
    for field in REVIEW_FIELDS:
        case[field] = ""
    case["groundedness_score"] = None
    return case


def load_insuranceqa() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in INSURANCEQA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda item: _stable_hash(f"{RANDOM_SEED}|{item[1]['question']}", 64))
    return [{"source_index": index, **row} for index, row in indexed]


def build_insuranceqa_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    pass_rows = rows[:110]

    for position, row in enumerate(pass_rows, start=1):
        qid = str(row["source_index"])
        answer = str(row["answer"]).strip()
        variant = "exact_answer"
        if position > 40 and position <= 80:
            answer = "Based on the supplied context, " + answer[0].lower() + answer[1:]
            variant = "supported_answer_preface"
        elif position > 80:
            answer = f"{answer} [insuranceqa:{qid}]"
            variant = "supported_answer_with_citation"
        cases.append(
            _new_case(
                case_id=f"ext-iqa-pass-{position:03d}",
                source_type="insuranceqa",
                source_file=str(INSURANCEQA_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                source_document_id=f"insuranceqa:{qid}",
                source_chunk_id=f"question_id:{qid}",
                question=str(row["question"]).strip(),
                context=[f"Question: {row['question']}\nAnswer: {row['answer']}"],
                candidate_answer=answer,
                generator_expected_label="PASS",
                mutation_type=variant,
                question_family_id=f"iqa-question:{qid}",
                document_family_id=f"iqa-document:{qid}",
                product_group=_product_group(row["question"]),
                base_leakage_group_id=f"iqa-family:{qid}",
                source_trace=f"JSONL row source_index={qid}",
            )
        )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    matrix = vectorizer.fit_transform([str(row["question"]) for row in rows])
    similarities = linear_kernel(matrix[:80], matrix)

    def distractor_for(target_pos: int, require_different_product: bool) -> dict[str, Any]:
        target = rows[target_pos]
        target_product = _product_group(target["question"])
        fallback: dict[str, Any] | None = None
        for index in np.argsort(-similarities[target_pos]):
            candidate = rows[int(index)]
            if candidate["source_index"] == target["source_index"]:
                continue
            if fallback is None:
                fallback = candidate
            different = _product_group(candidate["question"]) != target_product
            if different == require_different_product:
                return candidate
        if fallback is not None:
            return fallback
        raise RuntimeError("No InsuranceQA distractor found")

    failure_plan = (
        [("answer_from_distractor_document", False)] * 30
        + [("cross_product_confusion", True)] * 10
        + [("partially_supported_answer", False)] * 20
        + [("insufficient_evidence", False)] * 10
    )
    for position, (mutation, cross_product) in enumerate(failure_plan, start=1):
        target_pos = position - 1
        target = rows[target_pos]
        distractor = distractor_for(target_pos, cross_product)
        qid = str(target["source_index"])
        did = str(distractor["source_index"])
        if mutation == "partially_supported_answer":
            candidate_answer = (
                _first_sentence(str(target["answer"]))
                + " "
                + _first_sentence(str(distractor["answer"]))
            )
        elif mutation == "insufficient_evidence":
            candidate_answer = "The supplied context does not contain enough information to answer this question."
        else:
            candidate_answer = str(distractor["answer"]).strip()
        cases.append(
            _new_case(
                case_id=f"ext-iqa-fail-{position:03d}",
                source_type="insuranceqa",
                source_file=str(INSURANCEQA_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                source_document_id=f"insuranceqa:{qid}",
                source_chunk_id=f"question_id:{qid}",
                question=str(target["question"]).strip(),
                context=[f"Question: {target['question']}\nAnswer: {target['answer']}"],
                candidate_answer=candidate_answer,
                generator_expected_label="FAIL",
                mutation_type=mutation,
                question_family_id=f"iqa-question:{qid}",
                document_family_id=f"iqa-document:{qid}",
                product_group=_product_group(target["question"]),
                base_leakage_group_id=f"iqa-family:{qid}",
                hard_negative=True,
                mutation_source_file=str(INSURANCEQA_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                mutation_source_document_id=f"insuranceqa:{did}",
                mutation_detail=(
                    "Candidate answer is taken from a different InsuranceQA question; "
                    "for partially supported cases its first sentence follows a supported target sentence."
                    if mutation != "insufficient_evidence"
                    else "A false insufficiency response replaces the answer explicitly present in context."
                ),
                source_trace=f"target source_index={qid}; distractor source_index={did}",
            )
        )

    percent_rows = [
        row
        for row in rows[80:]
        if re.search(r"(?<!\d)(\d{1,3})\s*(?:%|percent)", str(row["answer"]), re.I)
    ][:10]
    if len(percent_rows) < 10:
        raise RuntimeError("Fewer than ten InsuranceQA percentage cases are available")
    for offset, row in enumerate(percent_rows, start=71):
        match = re.search(r"(?<!\d)(\d{1,3})(\s*(?:%|percent))", str(row["answer"]), re.I)
        assert match is not None
        original = int(match.group(1))
        replacement = (original + 10) if original <= 90 else (original - 10)
        mutated = str(row["answer"])[: match.start(1)] + str(replacement) + str(row["answer"])[match.end(1) :]
        qid = str(row["source_index"])
        cases.append(
            _new_case(
                case_id=f"ext-iqa-fail-{offset:03d}",
                source_type="insuranceqa",
                source_file=str(INSURANCEQA_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                source_document_id=f"insuranceqa:{qid}",
                source_chunk_id=f"question_id:{qid}",
                question=str(row["question"]).strip(),
                context=[f"Question: {row['question']}\nAnswer: {row['answer']}"],
                candidate_answer=mutated,
                generator_expected_label="FAIL",
                mutation_type="wrong_percentage",
                question_family_id=f"iqa-question:{qid}",
                document_family_id=f"iqa-document:{qid}",
                product_group=_product_group(row["question"]),
                base_leakage_group_id=f"iqa-family:{qid}",
                hard_negative=True,
                mutation_source_file=str(INSURANCEQA_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                mutation_source_document_id=f"insuranceqa:{qid}",
                mutation_detail=f"Changed the documented percentage from {original} to {replacement}.",
                source_trace=f"JSONL row source_index={qid}",
            )
        )
    return cases


@dataclass(frozen=True)
class PdfSentence:
    path: Path
    page_index: int
    sentence: str
    context: str
    product_group: str


def extract_pdf_sentences(paths: Sequence[Path]) -> list[PdfSentence]:
    records: list[PdfSentence] = []
    keyword = re.compile(
        r"\b(?:cover|covered|insurance|insured|exclude|excluded|benefit|deductible|"
        r"limit|maximum|condition|obligation|claim|damage|compensation|premium|liability)\w*\b",
        re.I,
    )
    for path in sorted(paths):
        reader = PdfReader(str(path))
        for page_index, page in enumerate(reader.pages):
            page_text = _clean_pdf_text(page.extract_text() or "")
            if len(page_text) < 120:
                continue
            for sentence in _sentences(page_text):
                if not keyword.search(sentence):
                    continue
                if sentence.count("|") > 2 or "Table of Contents" in sentence:
                    continue
                position = page_text.find(sentence)
                if position < 0:
                    context = page_text[:3000]
                else:
                    start = max(0, position - 850)
                    end = min(len(page_text), position + len(sentence) + 850)
                    context = page_text[start:end]
                records.append(
                    PdfSentence(
                        path=path,
                        page_index=page_index,
                        sentence=sentence,
                        context=context,
                        product_group=_product_group(path.stem + " " + sentence),
                    )
                )
    return records


def _round_robin(records: Sequence[PdfSentence], count: int) -> list[PdfSentence]:
    by_file: dict[Path, list[PdfSentence]] = defaultdict(list)
    for record in records:
        by_file[record.path].append(record)
    selected: list[PdfSentence] = []
    cursor = 0
    files = sorted(by_file)
    while len(selected) < count:
        made_progress = False
        for path in files:
            items = by_file[path]
            if cursor < len(items):
                selected.append(items[cursor])
                made_progress = True
                if len(selected) >= count:
                    break
        if not made_progress:
            break
        cursor += 1
    if len(selected) < count:
        raise RuntimeError(f"Only {len(selected)} suitable PDF sentences for requested {count}")
    return selected


def _pdf_case_common(record: PdfSentence) -> dict[str, Any]:
    relative = str(record.path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    document_id = record.path.stem
    return {
        "source_file": relative,
        "source_document_id": document_id,
        "source_page": record.page_index + 1,
        "source_chunk_id": f"page:{record.page_index + 1}",
        "question": f"According to {record.path.name}, what provision is stated in the supplied policy context?",
        "context": [record.context],
        "question_family_id": f"pdf-question:{document_id}:p{record.page_index + 1}",
        "document_family_id": f"pdf-document:{document_id}",
        "product_group": record.product_group,
        "base_leakage_group_id": f"pdf-family:{document_id}:p{record.page_index + 1}",
        "source_trace": f"{relative}, physical page {record.page_index + 1}",
    }


def build_helvetia_pdf_cases(active_records: list[PdfSentence]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    pass_records = _round_robin(active_records, 60)
    for position, record in enumerate(pass_records, start=1):
        common = _pdf_case_common(record)
        answer = record.sentence
        mutation = "exact_supported_excerpt"
        if position > 40:
            answer += f" [{record.path.name}, page {record.page_index + 1}]"
            mutation = "supported_excerpt_with_citation"
        cases.append(
            _new_case(
                case_id=f"ext-helvetia-pass-{position:03d}",
                source_type="helvetia_pdf",
                candidate_answer=answer,
                generator_expected_label="PASS",
                mutation_type=mutation,
                **common,
            )
        )

    numeric = [
        record
        for record in active_records
        if re.search(r"\b(?:maximum|limit|deductible|CHF|EUR|percent|days?|months?)\b", record.sentence, re.I)
        and re.search(r"(?<![A-Za-z])\d+(?:[.,]\d+)?", record.sentence)
    ]
    numeric = _round_robin(numeric, 20)
    donor_values = [
        re.search(r"(?<![A-Za-z])\d+(?:[.,]\d+)?", record.sentence).group(0)  # type: ignore[union-attr]
        for record in numeric
    ]
    for index, record in enumerate(numeric[:10], start=1):
        match = re.search(r"(?<![A-Za-z])\d+(?:[.,]\d+)?", record.sentence)
        assert match is not None
        replacement = donor_values[(index + 7) % len(donor_values)]
        if replacement == match.group(0):
            replacement = str(float(match.group(0).replace(",", ".")) + 1).rstrip("0").rstrip(".")
        mutated = record.sentence[: match.start()] + replacement + record.sentence[match.end() :]
        donor = numeric[(index + 7) % len(numeric)]
        cases.append(
            _new_case(
                case_id=f"ext-helvetia-fail-{index:03d}",
                source_type="helvetia_pdf",
                candidate_answer=mutated,
                generator_expected_label="FAIL",
                mutation_type="wrong_limit",
                hard_negative=True,
                mutation_source_file=str(donor.path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                mutation_source_document_id=donor.path.stem,
                mutation_source_page=donor.page_index + 1,
                mutation_detail=f"Changed the first documented numeric value from {match.group(0)} to {replacement}; replacement is traceable to another source excerpt.",
                **_pdf_case_common(record),
            )
        )

    polarity_records = [
        record
        for record in active_records
        if re.search(r"\b(?:is|are) covered\b|\bcovers\b|\bnot covered\b|\bexcluded\b", record.sentence, re.I)
    ]
    polarity_records = _round_robin(polarity_records, 10)
    for index, record in enumerate(polarity_records, start=11):
        sentence = record.sentence
        if re.search(r"\bnot covered\b", sentence, re.I):
            mutated = re.sub(r"\bnot covered\b", "covered", sentence, count=1, flags=re.I)
        elif re.search(r"\bexcluded\b", sentence, re.I):
            mutated = re.sub(r"\bexcluded\b", "included", sentence, count=1, flags=re.I)
        elif re.search(r"\b(is|are) covered\b", sentence, re.I):
            mutated = re.sub(r"\b(is|are) covered\b", lambda m: f"{m.group(1)} not covered", sentence, count=1, flags=re.I)
        else:
            mutated = re.sub(r"\bcovers\b", "does not cover", sentence, count=1, flags=re.I)
        cases.append(
            _new_case(
                case_id=f"ext-helvetia-fail-{index:03d}",
                source_type="helvetia_pdf",
                candidate_answer=mutated,
                generator_expected_label="FAIL",
                mutation_type="reversed_coverage",
                hard_negative=True,
                mutation_source_file=str(record.path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                mutation_source_document_id=record.path.stem,
                mutation_source_page=record.page_index + 1,
                mutation_detail="Mechanically reversed the first explicit coverage or exclusion predicate.",
                **_pdf_case_common(record),
            )
        )

    citation_records = _round_robin(active_records[60:], 10)
    for index, record in enumerate(citation_records, start=21):
        wrong_page = record.page_index + 2
        cases.append(
            _new_case(
                case_id=f"ext-helvetia-fail-{index:03d}",
                source_type="helvetia_pdf",
                candidate_answer=f"{record.sentence} [{record.path.name}, page {wrong_page}]",
                generator_expected_label="FAIL",
                mutation_type="citation_mismatch",
                hard_negative=True,
                mutation_source_file=str(record.path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                mutation_source_document_id=record.path.stem,
                mutation_source_page=wrong_page,
                mutation_detail=f"Claim text is copied from physical page {record.page_index + 1}, but the citation points to page {wrong_page}.",
                **_pdf_case_common(record),
            )
        )

    extra_records = _round_robin(active_records[100:], 20)
    for offset, record in enumerate(extra_records[:10], start=31):
        donor = next(
            candidate
            for candidate in extra_records[10:]
            if candidate.path != record.path
        )
        cases.append(
            _new_case(
                case_id=f"ext-helvetia-fail-{offset:03d}",
                source_type="helvetia_pdf",
                candidate_answer=f"{record.sentence} {donor.sentence}",
                generator_expected_label="FAIL",
                mutation_type="unsupported_extra_claim",
                hard_negative=True,
                mutation_source_file=str(donor.path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                mutation_source_document_id=donor.path.stem,
                mutation_source_page=donor.page_index + 1,
                mutation_detail="Appended a real but context-unsupported clause from a different active Helvetia document.",
                **_pdf_case_common(record),
            )
        )
    return cases


def build_baloise_hard_negatives(
    active_records: list[PdfSentence],
    excluded_records: list[PdfSentence],
) -> list[dict[str, Any]]:
    baloise = [record for record in excluded_records if record.path.name != "brochure-services (1).pdf"]
    donors = _round_robin(baloise, 20)
    active_sample = _round_robin(active_records, 80)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([record.sentence for record in active_sample + donors])
    similarities = linear_kernel(matrix[len(active_sample) :], matrix[: len(active_sample)])
    cases: list[dict[str, Any]] = []
    for index, donor in enumerate(donors, start=1):
        target = active_sample[int(np.argmax(similarities[index - 1]))]
        mutation = "old_document_instead_of_current" if index <= 10 else "wrong_exclusion"
        common = _pdf_case_common(target)
        # All Baloise versions of the same product family remain in one leakage group.
        common["base_leakage_group_id"] = f"excluded-baloise-family:{donor.product_group}"
        cases.append(
            _new_case(
                case_id=f"ext-baloise-fail-{index:03d}",
                source_type="baloise_excluded_pdf",
                candidate_answer=donor.sentence,
                generator_expected_label="FAIL",
                mutation_type=mutation,
                hard_negative=True,
                mutation_source_file=str(donor.path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                mutation_source_document_id=donor.path.stem,
                mutation_source_page=donor.page_index + 1,
                mutation_detail=(
                    "Candidate answer is copied from an excluded Baloise document while the context is an active Helvetia excerpt."
                    if mutation == "old_document_instead_of_current"
                    else "An exclusion/coverage statement from an excluded Baloise product is substituted for the active Helvetia context."
                ),
                source_trace=(
                    common["source_trace"]
                    + f"; excluded distractor {donor.path.relative_to(PROJECT_ROOT)}, physical page {donor.page_index + 1}"
                ),
                **{key: value for key, value in common.items() if key != "source_trace"},
            )
        )
    return cases


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _policy_context(policy: dict[str, str], contact: dict[str, str]) -> str:
    return (
        f"Customer: {contact['firstName']} {contact['lastName']} ({contact['emailAddress']}). "
        f"Policy number: {policy['policyNumber']}. Product type: {policy['productType']}. "
        f"Coverage type: {policy['coverageType']}. Status: {policy['status']}. "
        f"Start date: {policy['startDate']}. End date: {policy['endDate']}. "
        f"Deductible: {policy['deductible']} {policy['currency']}. "
        f"Annual premium: {policy['annualPremium']} {policy['currency']}."
    )


def _crm_common(policy: dict[str, str], contact: dict[str, str]) -> dict[str, Any]:
    email = contact["emailAddress"]
    return {
        "source_type": "synthetic_crm",
        "source_file": "data/synthetic/crm/policies.csv",
        "source_document_id": f"policy:{policy['policyNumber']}",
        "source_chunk_id": f"policyNumber:{policy['policyNumber']}",
        "context": [_policy_context(policy, contact)],
        "question_family_id": f"crm-policy-question:{policy['policyNumber']}",
        "document_family_id": f"crm-policy-document:{policy['policyNumber']}",
        "customer_group_id": f"customer:{email}",
        "policy_group_id": f"policy:{policy['policyNumber']}",
        "product_group": _product_group(policy["productType"]),
        "base_leakage_group_id": f"crm-customer-family:{email}",
        "source_trace": f"data/synthetic/crm/policies.csv policyNumber={policy['policyNumber']}; contacts.csv emailAddress={email}",
    }


def build_crm_cases() -> list[dict[str, Any]]:
    contacts = _read_csv(CRM_DIR / "contacts.csv")
    policies = _read_csv(CRM_DIR / "policies.csv")
    claims = _read_csv(CRM_DIR / "claims.csv")
    contact_by_email = {row["emailAddress"]: row for row in contacts}
    cases: list[dict[str, Any]] = []

    for index, policy in enumerate(policies, start=1):
        contact = contact_by_email[policy["customerEmail"]]
        common = _crm_common(policy, contact)
        answer = (
            f"{contact['firstName']} {contact['lastName']} holds policy {policy['policyNumber']}, "
            f"a {policy['productType']} policy with {policy['coverageType']}, status {policy['status']}, "
            f"a deductible of {policy['deductible']} {policy['currency']}, and an annual premium of "
            f"{policy['annualPremium']} {policy['currency']}."
        )
        cases.append(
            _new_case(
                case_id=f"ext-crm-pass-{index:03d}",
                question="What are the customer, policy number, product, coverage, status, deductible and annual premium?",
                candidate_answer=answer,
                generator_expected_label="PASS",
                mutation_type="supported_policy_multi_claim",
                **common,
            )
        )
    for offset, policy in enumerate(policies, start=17):
        contact = contact_by_email[policy["customerEmail"]]
        cases.append(
            _new_case(
                case_id=f"ext-crm-pass-{offset:03d}",
                question="What deductible and annual premium apply to this policy?",
                candidate_answer=f"The deductible is {policy['deductible']} {policy['currency']} and the annual premium is {policy['annualPremium']} {policy['currency']}.",
                generator_expected_label="PASS",
                mutation_type="supported_deductible_and_premium",
                **_crm_common(policy, contact),
            )
        )
    for offset, policy in enumerate(policies, start=33):
        contact = contact_by_email[policy["customerEmail"]]
        cases.append(
            _new_case(
                case_id=f"ext-crm-pass-{offset:03d}",
                question="What are the coverage type, policy status and contract dates?",
                candidate_answer=f"The coverage type is {policy['coverageType']}; the policy is {policy['status']} from {policy['startDate']} through {policy['endDate']}.",
                generator_expected_label="PASS",
                mutation_type="supported_coverage_status_dates",
                **_crm_common(policy, contact),
            )
        )
    lara = contact_by_email["lara.neumann@example.test"]
    lara_policies = [p for p in policies if p["customerEmail"] == lara["emailAddress"]]
    lara_context = [_policy_context(p, lara) for p in lara_policies]
    current = next(p for p in lara_policies if p["policyNumber"] == "TEST-KFZ-2026-1003")
    cases.append(
        _new_case(
            case_id="ext-crm-pass-049",
            source_type="synthetic_crm",
            source_file="data/synthetic/crm/policies.csv",
            source_document_id="customer:lara.neumann@example.test",
            source_chunk_id="customer_policy_set:lara.neumann@example.test",
            question=f"As of {REFERENCE_DATE}, what is Lara Neumann's current motor policy?",
            context=lara_context,
            candidate_answer=f"Lara Neumann's current motor policy is {current['policyNumber']} with {current['coverageType']}, a {current['deductible']} {current['currency']} deductible and a {current['annualPremium']} {current['currency']} annual premium.",
            generator_expected_label="PASS",
            mutation_type="supported_current_policy",
            question_family_id="crm-current-policy:lara-neumann",
            document_family_id="crm-policy-version-family:lara-motor",
            customer_group_id="customer:lara.neumann@example.test",
            policy_group_id="policy-version-family:lara-motor",
            product_group="motor",
            base_leakage_group_id="crm-customer-family:lara.neumann@example.test",
            source_trace="policies.csv: Lara Neumann policy set; reference date 2026-08-01",
        )
    )
    claim = claims[0]
    claim_policy = next(p for p in policies if p["policyNumber"] == claim["policyNumber"])
    claim_contact = contact_by_email[claim["customerEmail"]]
    claim_context = (
        f"Customer: {claim_contact['firstName']} {claim_contact['lastName']}. Claim number: {claim['claimNumber']}. "
        f"Policy number: {claim['policyNumber']}. Claim date: {claim['claimDate']}. Damage type: {claim['damageType']}. "
        f"Claimed amount: {claim['claimedAmount']} {claim['currency']}. Status: {claim['status']}."
    )
    cases.append(
        _new_case(
            case_id="ext-crm-pass-050",
            source_type="synthetic_crm",
            source_file="data/synthetic/crm/claims.csv",
            source_document_id=f"claim:{claim['claimNumber']}",
            source_chunk_id=f"claimNumber:{claim['claimNumber']}",
            question="What are the claim number, policy number, damage type, claimed amount and current status?",
            context=[claim_context],
            candidate_answer=f"Claim {claim['claimNumber']} concerns {claim['damageType']} under policy {claim['policyNumber']}, with {claim['claimedAmount']} {claim['currency']} claimed and status {claim['status']}.",
            generator_expected_label="PASS",
            mutation_type="supported_claim_record",
            question_family_id=f"crm-claim-question:{claim['claimNumber']}",
            document_family_id=f"crm-claim-document:{claim['claimNumber']}",
            customer_group_id=f"customer:{claim['customerEmail']}",
            policy_group_id=f"policy:{claim_policy['policyNumber']}",
            product_group=_product_group(claim_policy["productType"]),
            base_leakage_group_id=f"crm-customer-family:{claim['customerEmail']}",
            source_trace=f"claims.csv claimNumber={claim['claimNumber']}",
        )
    )

    def different_policy(target: dict[str, str], field: str) -> dict[str, str]:
        for candidate in policies:
            if candidate["policyNumber"] != target["policyNumber"] and candidate[field] != target[field]:
                return candidate
        raise RuntimeError(f"No different policy for field {field}")

    category_specs = [
        ("wrong_customer", 10),
        ("wrong_policy", 10),
        ("wrong_policy_status", 10),
        ("wrong_deductible", 10),
        ("wrong_premium", 10),
        ("wrong_coverage", 10),
        ("wrong_date", 10),
        ("correct_fact_but_wrong_entity", 10),
        ("correct_number_but_wrong_policy", 10),
    ]
    fail_counter = 0
    for mutation, count in category_specs:
        for local_index in range(count):
            fail_counter += 1
            target = policies[(fail_counter - 1) % len(policies)]
            contact = contact_by_email[target["customerEmail"]]
            common = _crm_common(target, contact)
            if mutation == "wrong_customer":
                donor = next(c for c in contacts if c["emailAddress"] != contact["emailAddress"])
                question = "Which customer holds the policy shown in the context?"
                answer = f"The policy belongs to {donor['firstName']} {donor['lastName']}."
                detail = f"Replaced {contact['firstName']} {contact['lastName']} with {donor['firstName']} {donor['lastName']} from contacts.csv."
                donor_id = donor["emailAddress"]
            elif mutation == "wrong_policy":
                donor_policy = different_policy(target, "policyNumber")
                question = "What is the policy number?"
                answer = f"The policy number is {donor_policy['policyNumber']}."
                detail = f"Substituted policy {target['policyNumber']} with real distractor {donor_policy['policyNumber']}."
                donor_id = donor_policy["policyNumber"]
            elif mutation == "wrong_policy_status":
                donor_policy = different_policy(target, "status")
                question = "What is the current policy status?"
                answer = f"The policy status is {donor_policy['status']}."
                detail = f"Substituted status {target['status']} with {donor_policy['status']} from {donor_policy['policyNumber']}."
                donor_id = donor_policy["policyNumber"]
            elif mutation == "wrong_deductible":
                donor_policy = different_policy(target, "deductible")
                question = "What deductible applies to this policy?"
                answer = f"The deductible is {donor_policy['deductible']} {donor_policy['currency']}."
                detail = f"Substituted deductible {target['deductible']} with {donor_policy['deductible']} from {donor_policy['policyNumber']}."
                donor_id = donor_policy["policyNumber"]
            elif mutation == "wrong_premium":
                donor_policy = different_policy(target, "annualPremium")
                question = "What is the annual premium of this policy?"
                answer = f"The annual premium is {donor_policy['annualPremium']} {donor_policy['currency']}."
                detail = f"Substituted annual premium {target['annualPremium']} with {donor_policy['annualPremium']} from {donor_policy['policyNumber']}"
                donor_id = donor_policy["policyNumber"]
            elif mutation == "wrong_coverage":
                donor_policy = different_policy(target, "coverageType")
                question = "What coverage type applies to this policy?"
                answer = f"The coverage type is {donor_policy['coverageType']}."
                detail = f"Substituted coverage {target['coverageType']} with {donor_policy['coverageType']} from {donor_policy['policyNumber']}."
                donor_id = donor_policy["policyNumber"]
            elif mutation == "wrong_date":
                donor_policy = different_policy(target, "startDate")
                question = "What are the start and end dates of this policy?"
                answer = f"The policy runs from {donor_policy['startDate']} through {donor_policy['endDate']}."
                detail = f"Substituted contract dates with dates from {donor_policy['policyNumber']}."
                donor_id = donor_policy["policyNumber"]
            elif mutation == "correct_fact_but_wrong_entity":
                donor_policy = next(p for p in policies if p["customerEmail"] != target["customerEmail"])
                donor_contact = contact_by_email[donor_policy["customerEmail"]]
                question = "Who holds this policy and what is its annual premium?"
                answer = f"{donor_contact['firstName']} {donor_contact['lastName']} holds it, and the annual premium is {target['annualPremium']} {target['currency']}."
                detail = f"Kept the target premium but assigned it to {donor_contact['firstName']} {donor_contact['lastName']}."
                donor_id = donor_contact["emailAddress"]
            else:
                donor_policy = different_policy(target, "annualPremium")
                question = "Which policy has the annual premium shown in the context?"
                answer = f"The annual premium of {target['annualPremium']} {target['currency']} belongs to policy {donor_policy['policyNumber']}."
                detail = f"Kept the target premium but attributed it to {donor_policy['policyNumber']}."
                donor_id = donor_policy["policyNumber"]
            cases.append(
                _new_case(
                    case_id=f"ext-crm-fail-{fail_counter:03d}",
                    question=question,
                    candidate_answer=answer,
                    generator_expected_label="FAIL",
                    mutation_type=mutation,
                    hard_negative=True,
                    mutation_source_file="data/synthetic/crm/policies.csv" if "customer" not in mutation else "data/synthetic/crm/contacts.csv",
                    mutation_source_document_id=str(donor_id),
                    mutation_detail=detail,
                    **common,
                )
            )

    old = next(p for p in lara_policies if p["policyNumber"] == "TEST-KFZ-2026-1001")
    for local_index in range(10):
        fail_counter += 1
        variants = [
            f"Lara Neumann's current motor policy is {old['policyNumber']} with a {old['deductible']} {old['currency']} deductible and a {old['annualPremium']} {old['currency']} annual premium.",
            f"The current motor contract is {old['policyNumber']}.",
            f"The deductible of Lara Neumann's current motor policy is {old['deductible']} {old['currency']}.",
            f"The annual premium of Lara Neumann's current motor policy is {old['annualPremium']} {old['currency']}.",
            f"As of {REFERENCE_DATE}, policy {old['policyNumber']} is the current motor policy.",
        ]
        cases.append(
            _new_case(
                case_id=f"ext-crm-fail-{fail_counter:03d}",
                source_type="synthetic_crm",
                source_file="data/synthetic/crm/policies.csv",
                source_document_id="customer:lara.neumann@example.test",
                source_chunk_id="customer_policy_set:lara.neumann@example.test",
                question=f"As of {REFERENCE_DATE}, what is Lara Neumann's current motor policy and its applicable details?",
                context=lara_context,
                candidate_answer=variants[local_index % len(variants)],
                generator_expected_label="FAIL",
                mutation_type="wrong_current_policy",
                question_family_id="crm-current-policy:lara-neumann",
                document_family_id="crm-policy-version-family:lara-motor",
                customer_group_id="customer:lara.neumann@example.test",
                policy_group_id="policy-version-family:lara-motor",
                product_group="motor",
                base_leakage_group_id="crm-customer-family:lara.neumann@example.test",
                hard_negative=True,
                mutation_source_file="data/synthetic/crm/policies.csv",
                mutation_source_document_id=f"policy:{old['policyNumber']}",
                mutation_detail=f"Selected older policy {old['policyNumber']} instead of later-starting policy {current['policyNumber']} for reference date {REFERENCE_DATE}.",
                source_trace="policies.csv: all Lara Neumann policies; current-policy rule uses active interval then latest start date",
            )
        )

    for local_index in range(10):
        fail_counter += 1
        target_claim = claims[local_index % len(claims)]
        target_policy = next(p for p in policies if p["policyNumber"] == target_claim["policyNumber"])
        target_contact = contact_by_email[target_claim["customerEmail"]]
        incorrect_status = "Approved" if target_claim["status"] != "Approved" else "Rejected"
        context = (
            f"Customer: {target_contact['firstName']} {target_contact['lastName']}. Claim number: {target_claim['claimNumber']}. "
            f"Policy number: {target_claim['policyNumber']}. Damage type: {target_claim['damageType']}. "
            f"Claimed amount: {target_claim['claimedAmount']} {target_claim['currency']}. Current status: {target_claim['status']}."
        )
        cases.append(
            _new_case(
                case_id=f"ext-crm-fail-{fail_counter:03d}",
                source_type="synthetic_crm",
                source_file="data/synthetic/crm/claims.csv",
                source_document_id=f"claim:{target_claim['claimNumber']}",
                source_chunk_id=f"claimNumber:{target_claim['claimNumber']}",
                question="Has this claim been approved or rejected according to the stored claim record?",
                context=[context],
                candidate_answer=f"The claim decision is {incorrect_status}.",
                generator_expected_label="FAIL",
                mutation_type="unsupported_claim_decision",
                question_family_id=f"crm-claim-question:{target_claim['claimNumber']}",
                document_family_id=f"crm-claim-document:{target_claim['claimNumber']}",
                customer_group_id=f"customer:{target_claim['customerEmail']}",
                policy_group_id=f"policy:{target_policy['policyNumber']}",
                product_group=_product_group(target_policy["productType"]),
                base_leakage_group_id=f"crm-customer-family:{target_claim['customerEmail']}",
                hard_negative=True,
                mutation_source_file="data/synthetic/crm/claims.csv",
                mutation_source_document_id=f"claim:{target_claim['claimNumber']}",
                mutation_detail=f"Changed stored claim status {target_claim['status']} to {incorrect_status}; no new claim decision was made.",
                source_trace=f"claims.csv claimNumber={target_claim['claimNumber']}",
            )
        )
    if len([case for case in cases if case["generator_expected_label"] == "PASS"]) != 50:
        raise AssertionError("CRM PASS target count drifted")
    if len([case for case in cases if case["generator_expected_label"] == "FAIL"]) != 110:
        raise AssertionError("CRM FAIL target count drifted")
    return cases


def _legacy_mutation(case_id: str, expected: str) -> str:
    if expected == "PASS":
        return "legacy_supported_candidate"
    mapping = {
        "deductible": "wrong_deductible",
        "contract_identifier": "wrong_policy",
        "coverage_type": "wrong_coverage",
        "reversed_coverage": "reversed_coverage",
        "zero_deductible": "wrong_deductible",
        "rental_car": "unsupported_extra_claim",
        "policy_dates": "wrong_date",
        "home_deductible": "wrong_deductible",
        "baggage_limit": "wrong_limit",
        "dental_percentage": "wrong_percentage",
        "mixed_extra_claim": "partially_supported_answer",
        "cancellation": "wrong_policy_status",
        "liability_distractor": "answer_from_distractor_document",
        "information_missing": "insufficient_evidence",
    }
    for token, mutation in mapping.items():
        if token in case_id:
            return mutation
    return "other"


def build_legacy_cases() -> list[dict[str, Any]]:
    payload = json.loads(LEGACY_PATH.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    for position, source in enumerate(payload["cases"], start=1):
        expected = "PASS" if bool(source["label"]) else "FAIL"
        identifier = str(source["id"])
        cases.append(
            _new_case(
                case_id=f"ext-legacy-{position:03d}",
                source_type="legacy_synthetic",
                source_file=str(LEGACY_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                source_document_id=f"legacy:{identifier}",
                source_chunk_id=identifier,
                question=str(source["query"]),
                context=[str(item) for item in source["documents"]],
                candidate_answer=str(source["answer"]),
                generator_expected_label=expected,
                mutation_type=_legacy_mutation(identifier, expected),
                question_family_id=f"legacy-question:{identifier}",
                document_family_id=f"legacy-document:{identifier}",
                product_group=_product_group(source["query"]),
                base_leakage_group_id=f"legacy-family:{identifier}",
                hard_negative=expected == "FAIL",
                mutation_source_file=str(LEGACY_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                mutation_source_document_id=f"legacy:{identifier}",
                mutation_detail="Existing synthetic fixture label retained only as generator expectation; human re-review required.",
                source_trace=f"groundedness_calibration_cases.json case id={identifier}",
            )
        )
    return cases


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            smaller, larger = sorted((root_left, root_right))
            self.parent[larger] = smaller


def find_near_duplicates(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    texts = [
        _normalize(case["question"] + " " + case["candidate_answer"])
        for case in cases
    ]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(8, len(cases)))
    neighbors.fit(matrix)
    distances, indices = neighbors.kneighbors(matrix)
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for left, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        for distance, right in zip(row_distances[1:], row_indices[1:]):
            right = int(right)
            key = tuple(sorted((left, right)))
            similarity = 1.0 - float(distance)
            if key in seen or similarity < NEAR_DUPLICATE_THRESHOLD:
                continue
            seen.add(key)
            pairs.append(
                {
                    "case_id_a": cases[left]["case_id"],
                    "case_id_b": cases[right]["case_id"],
                    "similarity": round(similarity, 6),
                    "base_group_a": cases[left]["base_leakage_group_id"],
                    "base_group_b": cases[right]["base_leakage_group_id"],
                }
            )
    return pairs


def merge_leakage_groups(cases: list[dict[str, Any]], near_pairs: Sequence[dict[str, Any]]) -> None:
    groups = {str(case["base_leakage_group_id"]) for case in cases}
    union = UnionFind(groups)
    by_id = {str(case["case_id"]): case for case in cases}
    # Explicit family relationships take precedence over row-level splitting.
    # This joins PASS/FAIL variants, chunks from the same document, customer
    # records, and old/current policies before similarity-based unions.
    for field in (
        "question_family_id",
        "document_family_id",
        "customer_group_id",
        "policy_group_id",
    ):
        first_group_by_value: dict[str, str] = {}
        for case in cases:
            value = str(case.get(field) or "").strip()
            if not value:
                continue
            group = str(case["base_leakage_group_id"])
            if value in first_group_by_value:
                union.union(first_group_by_value[value], group)
            else:
                first_group_by_value[value] = group
    for pair in near_pairs:
        left = by_id[str(pair["case_id_a"])]["base_leakage_group_id"]
        right = by_id[str(pair["case_id_b"])]["base_leakage_group_id"]
        union.union(str(left), str(right))
    components: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        components[union.find(group)].append(group)
    resolved = {
        member: f"lg-{_stable_hash('|'.join(sorted(members)))}"
        for members in components.values()
        for member in members
    }
    for case in cases:
        case["leakage_group_id"] = resolved[str(case["base_leakage_group_id"])]


def assign_group_splits(cases: list[dict[str, Any]]) -> dict[str, Any]:
    fractions = {
        "CALIBRATION": 0.55,
        "VALIDATION": 0.18,
        "LOCKED_HOLDOUT_CANDIDATE": 0.27,
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[str(case["leakage_group_id"])].append(case)
    totals = {
        "total": len(cases),
        "PASS": sum(case["generator_expected_label"] == "PASS" for case in cases),
        "FAIL": sum(case["generator_expected_label"] == "FAIL" for case in cases),
    }
    target = {
        split: {key: fractions[split] * value for key, value in totals.items()}
        for split in fractions
    }
    state = {split: {"total": 0, "PASS": 0, "FAIL": 0} for split in fractions}

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), _stable_hash(f"{RANDOM_SEED}|{item[0]}", 64)),
    )
    assignments: dict[str, str] = {}
    for group_id, members in ordered_groups:
        contribution = {
            "total": len(members),
            "PASS": sum(case["generator_expected_label"] == "PASS" for case in members),
            "FAIL": sum(case["generator_expected_label"] == "FAIL" for case in members),
        }
        def global_cost(candidate_split: str) -> float:
            cost = 0.0
            for split in fractions:
                for key in ("total", "PASS", "FAIL"):
                    projected = state[split][key]
                    if split == candidate_split:
                        projected += contribution[key]
                    # Normalize by the corpus total rather than the split target.
                    # Otherwise the smallest split is systematically filled first.
                    delta = (projected - target[split][key]) / max(totals[key], 1.0)
                    cost += delta * delta
                    if projected > target[split][key]:
                        cost += 2.0 * delta * delta
            return cost

        best_split = min(fractions, key=global_cost)
        assignments[group_id] = best_split
        for key in state[best_split]:
            state[best_split][key] += contribution[key]
    for case in cases:
        case["split"] = assignments[str(case["leakage_group_id"])]
    return {
        "target_fractions": fractions,
        "actual_generator_expectation_counts": state,
        "group_count": len(groups),
    }


def _probably_english(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    latin = sum("a" <= char.casefold() <= "z" for char in letters)
    return latin / len(letters) >= 0.85


def quality_checks(
    cases: list[dict[str, Any]],
    near_pairs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    required = [
        "case_id", "source_type", "source_file", "source_document_id", "question",
        "context", "candidate_answer", "generator_expected_label", "mutation_type",
        "question_family_id", "document_family_id", "product_group", "annotation_status",
        "split", "leakage_group_id",
    ]
    id_counts = Counter(str(case["case_id"]) for case in cases)
    tuple_counts = Counter(
        (
            _normalize(str(case["question"])),
            _normalize(" ".join(case["context"])),
            _normalize(str(case["candidate_answer"])),
        )
        for case in cases
    )
    answer_labels: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        answer_labels[_normalize(str(case["candidate_answer"]))].add(str(case["generator_expected_label"]))

    invalid_sources: list[str] = []
    flag_counts: Counter[str] = Counter()
    for case in cases:
        flags: list[str] = []
        missing = [field for field in required if case.get(field) in (None, "", [])]
        if missing:
            flags.append("missing_required_fields:" + ",".join(missing))
        if id_counts[str(case["case_id"])] > 1:
            flags.append("duplicate_case_id")
        key = (
            _normalize(str(case["question"])),
            _normalize(" ".join(case["context"])),
            _normalize(str(case["candidate_answer"])),
        )
        if tuple_counts[key] > 1:
            flags.append("exact_duplicate_triplet")
        if len(answer_labels[_normalize(str(case["candidate_answer"]))]) > 1:
            flags.append("identical_answer_across_expected_labels")
        source = PROJECT_ROOT / str(case["source_file"])
        if not source.exists():
            flags.append("invalid_source_reference")
            invalid_sources.append(str(case["case_id"]))
        if case.get("mutation_source_file") and not (PROJECT_ROOT / str(case["mutation_source_file"])).exists():
            flags.append("invalid_mutation_source_reference")
            invalid_sources.append(str(case["case_id"]))
        context_text = "\n".join(str(item) for item in case["context"])
        if len(context_text) > MAX_CONTEXT_CHARS:
            flags.append("context_too_long")
        if not _probably_english(str(case["question"]) + " " + str(case["candidate_answer"])):
            flags.append("language_review_required")
        if "�" in context_text or "�" in str(case["candidate_answer"]):
            flags.append("unreadable_replacement_character")
        if case["generator_expected_label"] == "FAIL":
            answer_tokens = set(_normalize(str(case["candidate_answer"])).split())
            context_tokens = set(_normalize(context_text).split())
            overlap = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
            if overlap < 0.08:
                flags.append("possibly_trivial_fail_low_lexical_overlap")
            if not case.get("mutation_detail"):
                flags.append("mutation_trace_missing")
        if flags:
            case["annotation_status"] = "REVIEW_REQUIRED"
        case["quality_flags"] = flags
        flag_counts.update(flags)

    split_by_group: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        split_by_group[str(case["leakage_group_id"])].add(str(case["split"]))
    leakage_violations = {
        group: sorted(splits)
        for group, splits in split_by_group.items()
        if len(splits) > 1
    }
    by_id = {str(case["case_id"]): case for case in cases}
    cross_split_near_pairs = [
        pair
        for pair in near_pairs
        if by_id[str(pair["case_id_a"])]["split"] != by_id[str(pair["case_id_b"])]["split"]
    ]
    return {
        "flag_counts": dict(flag_counts),
        "flagged_cases": sum(bool(case["quality_flags"]) for case in cases),
        "exact_duplicate_triplets": sum(count - 1 for count in tuple_counts.values() if count > 1),
        "duplicate_case_ids": sum(count - 1 for count in id_counts.values() if count > 1),
        "identical_answers_across_expected_labels": sum(len(labels) > 1 for labels in answer_labels.values()),
        "near_duplicate_pairs": len(near_pairs),
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "invalid_source_cases": sorted(set(invalid_sources)),
        "leakage_group_split_violations": leakage_violations,
        "cross_split_near_duplicate_pairs": cross_split_near_pairs,
        "excluded_cases": sum(bool(case["excluded"]) for case in cases),
    }


def _count_rows(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows).items()))


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return "\n---\n".join(str(item) for item in value)
    if value is None:
        return ""
    return value


def write_review_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    # The reviewer-facing file is deliberately blind: technical expectations,
    # mutation provenance, hard-negative flags, and groundedness scores stay only
    # in the candidate/audit artifacts until independent annotation is complete.
    fieldnames = [
        "case_id", "source_type", "source_file", "source_document_id", "source_page",
        "source_chunk_id", "question", "context", "candidate_answer",
        "question_family_id", "document_family_id", "customer_group_id",
        "policy_group_id", "product_group", *REVIEW_FIELDS,
        "annotation_status", "split", "leakage_group_id", "language",
        "source_trace", "quality_flags", "excluded", "exclusion_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def source_audit_markdown(
    *,
    insuranceqa_rows: Sequence[dict[str, Any]],
    active_pdf_records: Sequence[PdfSentence],
    excluded_pdf_records: Sequence[PdfSentence],
) -> str:
    contacts = _read_csv(CRM_DIR / "contacts.csv")
    policies = _read_csv(CRM_DIR / "policies.csv")
    claims = _read_csv(CRM_DIR / "claims.csv")
    legacy = json.loads(LEGACY_PATH.read_text(encoding="utf-8"))["cases"]
    active_pdfs = sorted(HELvetia_DIR.glob("*.pdf"))
    excluded_pdfs = sorted(EXCLUDED_PDF_DIR.glob("*.pdf"))
    active_pages = sum(len(PdfReader(str(path)).pages) for path in active_pdfs)
    excluded_pages = sum(len(PdfReader(str(path)).pages) for path in excluded_pdfs)
    return f"""# Data Source Audit - Extended Groundedness Dataset

Generated: `{datetime.now(timezone.utc).isoformat()}`

This audit was completed read-only before candidate artifacts were generated.
No Chroma retrieval, collection mutation, reindex, embedding generation, CRM
write, LLM call, Self-Check or `/api/ask` call was performed.

| Data source | Exact path | Format / available entries | Fields / language | PASS use | FAIL / hard-negative use | Quality and leakage risks |
|---|---|---|---|---|---|---|
| InsuranceQA local benchmark | `data/benchmarks/qa/insuranceqa/data_insuranceqa_1000.jsonl` | JSONL, {len(insuranceqa_rows)} unique Q/A rows | `question`, `answer`; English | Exact sourced answers, supported prefacing, valid citations | Similar-question distractors, cross-product answers, mixed supported/unsupported answers, wrong percentages | Public/general-domain answers can be dated or informal; similar questions must stay grouped; no human Groundedness annotation |
| InsuranceQA thesis subset | `data/benchmarks/qa/insuranceqa/data_insuranceqa_thesis_200.jsonl` | JSONL, 200 rows, complete subset overlap with the 1,000-row file | Q/A plus evaluation metadata; English | Audit/reference only | Audit/reference only | Not independent from the 1,000-row source; must never be split as a separate hold-out |
| InsuranceQA Chroma | `data/processed/vectorstores/chroma_db/chroma.sqlite3`, collection `insuranceqa_collection` | Read-only collection metadata, 1,248 chunks, dimension 1,024 | `source`, `question_id`, `dataset`, `start_index`; English content | Confirms indexed representation | Not used to generate candidates | Chunks from one Q/A can leak across splits; collection was inspected only by SQLite counts |
| Active Helvetia PDFs | `data/raw/pdfs/*.pdf` | {len(active_pdfs)} PDFs, {active_pages} pages, 1,585 indexed chunks; {len(active_pdf_records)} usable sentence candidates | English policy/product text and page references | Exact provisions, supported citations, multi-clause contexts | Numeric mutations, polarity reversal, citation mismatch, unsupported extra clauses | PDF extraction can join headers, tables or hyphenated text; all cases require visual/source review; multiple pages of one document are related |
| Excluded Baloise/duplicate PDFs | `data/raw/excluded_pdfs/*.pdf` | {len(excluded_pdfs)} PDFs, {excluded_pages} pages; {len(excluded_pdf_records)} usable sentence candidates | Three English Baloise products plus one duplicate Helvetia brochure | Not used as current-product PASS evidence | Explicit old-document and cross-provider distractors only | Not part of the current Helvetia corpus; high risk of accidental provider/version leakage; kept in source-family groups |
| Synthetic CRM contacts | `data/synthetic/crm/contacts.csv` | CSV, {len(contacts)} contacts | name and test email; English-compatible synthetic values | Correct customer association | Wrong-customer/entity substitutions | Synthetic only; same customer and all related records must share a split |
| Synthetic CRM policies | `data/synthetic/crm/policies.csv` | CSV, {len(policies)} policies | policy number, product, coverage, status, dates, deductible, premium, currency, customer | Exact policy facts and multi-claim answers | Wrong policy/current policy/status/deductible/premium/coverage/date/entity | Small number of customers; Lara current-policy family is one concentrated template family |
| Synthetic CRM claims | `data/synthetic/crm/claims.csv` | CSV, {len(claims)} claims | claim/policy/customer relationship, date, type, amount, status | Exact claim facts | Unsupported claim decisions | Status is a stored fact, not a legal claim decision; policy/customer groups must stay together |
| CRM schema and seed workflow | `data/synthetic/crm/schema.json`; `scripts/seed_espocrm.py` | JSON schema plus Python seed script | Contact, MvaPolicy, MvaClaim relationships | Provenance and relationship validation | Relationship hard negatives | Seed script is not called; live CRM is not used or modified |
| Legacy groundedness fixture | `tests/fixtures/groundedness_calibration_cases.json` | JSON, {len(legacy)} synthetic cases (14 expected PASS, 14 expected FAIL) | query, documents, answer, boolean technical label; English | Re-review existing supported candidates | Re-review existing mutated candidates | Existing labels have no documented human provenance; same cases created and evaluated old threshold 0.51 |
| Existing score artifact | `config/groundedness_calibration.json` | JSON, 28 stored scores; threshold 0.51 | `fact_aware_claim_support_v4` scores | Score reproducibility audit only | Score reproducibility audit only | Must not be visible during annotation and is not Ground Truth |
| Reranker fixtures | `tests/fixtures/reranker_*cases.json` | JSON, 6 + 64 + 64 ranking cases | query, candidates, correct candidate ID | Scenario inventory | Hard-negative pattern inventory | Ranking labels are not answer-groundedness labels; no automatic transfer |

## Implementation paths

- `src/data/insuranceqa_ingestion.py:43, 75-123, 138-245`: local/HuggingFace loading, Q/A document mapping, chunking and separate collection.
- `scripts/seed_espocrm.py:143-227`: reads the three synthetic CSVs and seeds relationships; not executed here.
- `src/guardrails/integrations/nemo_actions.py:178, 345-493`: deterministic `fact_aware_claim_support_v4` implementation.
- `scripts/calibrate_groundedness.py:46-115`: prior 28-case score/threshold workflow.

## Audit conclusion

The sources are sufficient to prepare 400-500 review candidates without
inventing human labels. They are not sufficient for a final calibrated
threshold until two independent reviewers annotate labels, error categories
and criticality and all conflicts are adjudicated.
"""


def dictionary_markdown() -> str:
    definitions = {
        "case_id": "Stable unique candidate identifier.",
        "source_type": "insuranceqa, helvetia_pdf, baloise_excluded_pdf, synthetic_crm, or legacy_synthetic.",
        "source_file": "Repository-relative primary evidence path.",
        "source_document_id": "Stable source record or document identifier.",
        "source_page": "Physical one-based PDF page where applicable.",
        "source_chunk_id": "Source-specific chunk/row identifier.",
        "question": "English evaluation question.",
        "context": "Stored evidence supplied to the evaluator; list in JSONL, joined in CSV/XLSX.",
        "candidate_answer": "English answer independently judged by reviewers.",
        "generator_expected_label": "Technical construction expectation only; never Ground Truth and hidden from reviewer sheets.",
        "mutation_type": "Reproducible technical case construction or hard-negative mechanism.",
        "question_family_id": "Groups paraphrases/variants of one source question.",
        "document_family_id": "Groups records from the same source document/version family.",
        "customer_group_id": "Synthetic customer relationship group, blank for non-CRM cases.",
        "policy_group_id": "Synthetic policy/version group, blank for non-CRM cases.",
        "product_group": "Coarse insurance product stratum.",
        "reviewer_1_label": "Independent reviewer 1 label: PASS, FAIL, AMBIGUOUS, EXCLUDE.",
        "reviewer_2_label": "Independent reviewer 2 label: PASS, FAIL, AMBIGUOUS, EXCLUDE.",
        "adjudicated_label": "Final adjudicated label; remains blank until conflicts are resolved.",
        "reviewer_*_error_category": "Reviewer-selected error category; use none for PASS and other with explanation when needed.",
        "adjudicated_error_category": "Final adjudicated category.",
        "reviewer_*_criticality": "CRITICAL, NON_CRITICAL, or REVIEW_REQUIRED.",
        "adjudicated_criticality": "Final adjudicated criticality.",
        "reviewer_*_notes": "Evidence-based rationale; no score information is shown.",
        "adjudication_notes": "Resolution rationale for reviewer conflict.",
        "reviewer_*_confidence": "Integer confidence from 1 (very uncertain) to 5 (highly confident).",
        "annotation_status": "PENDING, REVIEW_REQUIRED, REVIEWED, CONFLICT, or ADJUDICATED.",
        "groundedness_score": "Blank during review; populated only by isolated post-annotation scoring.",
        "split": "Preliminary CALIBRATION, VALIDATION, or LOCKED_HOLDOUT_CANDIDATE group assignment.",
        "leakage_group_id": "All related or near-duplicate candidates share one split through this ID.",
        "quality_flags": "Non-destructive technical warnings requiring review.",
        "hard_negative": "True when a candidate intentionally uses a plausible but technically mutated answer.",
    }
    rows = "\n".join(f"| `{field}` | {definition} |" for field, definition in definitions.items())
    return f"""# Groundedness Extended Dataset - Data Dictionary

| Field | Definition |
|---|---|
{rows}

## Human label definitions

- **PASS:** Every material answer claim is supported by context, belongs to the correct entity and answers the question.
- **FAIL:** At least one material claim is unsupported, contradicted, attached to the wrong entity or technically incorrect.
- **AMBIGUOUS:** Question, context or answer does not permit a unique expert judgment.
- **EXCLUDE:** The case is technically defective, duplicate or unsuitable for the study.

`generator_expected_label` must not be copied into any reviewer or adjudicated
field. Groundedness scores remain unavailable until annotation and adjudication
are complete.
"""


def quality_markdown(cases: Sequence[dict[str, Any]], quality: dict[str, Any]) -> str:
    flags = "\n".join(
        f"| `{name}` | {count} |" for name, count in sorted(quality["flag_counts"].items())
    ) or "| none | 0 |"
    return f"""# Extended Groundedness Dataset - Technical Quality Report

## Result

- Candidates checked: **{len(cases)}**
- Cases with at least one non-destructive quality flag: **{quality['flagged_cases']}**
- Duplicate case IDs: **{quality['duplicate_case_ids']}**
- Exact duplicate triplets: **{quality['exact_duplicate_triplets']}**
- Identical answers across technical PASS/FAIL expectations: **{quality['identical_answers_across_expected_labels']}**
- Near-duplicate pairs at cosine similarity >= {NEAR_DUPLICATE_THRESHOLD:.2f}: **{quality['near_duplicate_pairs']}**
- Invalid source references: **{len(quality['invalid_source_cases'])}**
- Contexts longer than {MAX_CONTEXT_CHARS} characters: **{quality['flag_counts'].get('context_too_long', 0)}**
- Cases excluded automatically: **{quality['excluded_cases']}**

No candidate was silently deleted. Flagged cases remain in the review dataset
with `annotation_status=REVIEW_REQUIRED`.

## Quality flags

| Flag | Count |
|---|---:|
{flags}

## Methods

- Exact duplicates: normalized `(question, context, candidate_answer)` equality.
- Near duplicates: word unigram/bigram TF-IDF cosine similarity, threshold 0.94.
- English check: conservative Latin-character heuristic; uncertain cases are flagged, not removed.
- Source validation: every primary and mutation-source path must exist under the repository.
- Mutation trace: every technical FAIL must include a documented mutation detail.
- Trivial FAIL screen: lexical answer/context overlap below 0.08 is flagged for human review.
"""


def leakage_markdown(
    cases: Sequence[dict[str, Any]],
    near_pairs: Sequence[dict[str, Any]],
    quality: dict[str, Any],
    split_plan: dict[str, Any],
) -> str:
    split_counts = Counter(str(case["split"]) for case in cases)
    return f"""# Extended Groundedness Dataset - Duplicate and Leakage Report

## Grouping method

Initial families were defined from source question, document/page, customer,
policy/version and synthetic template relationships. Near-duplicate pairs were
then detected with TF-IDF word unigram/bigram cosine similarity >=
{NEAR_DUPLICATE_THRESHOLD:.2f}. Any connected initial families were unioned
before split assignment. No case-level random split was used.

## Results

- Final leakage groups: **{split_plan['group_count']}**
- Near-duplicate pairs: **{len(near_pairs)}**
- Leakage groups appearing in multiple splits: **{len(quality['leakage_group_split_violations'])}**
- Near-duplicate pairs crossing splits: **{len(quality['cross_split_near_duplicate_pairs'])}**
- Calibration candidates: **{split_counts.get('CALIBRATION', 0)}**
- Validation candidates: **{split_counts.get('VALIDATION', 0)}**
- Locked hold-out candidates: **{split_counts.get('LOCKED_HOLDOUT_CANDIDATE', 0)}**

The hold-out assignment is preliminary and must not be finalized until human
annotation is complete. Generator expectations were used only to balance this
planning split; they are not Ground Truth.

## Known leakage risks

1. InsuranceQA questions can be semantically related even below the 0.94 threshold.
2. CRM variants for one customer may be numerous; all are kept in one customer group.
3. All Lara old/current-policy variants are kept together.
4. Active Helvetia and excluded Baloise source families are explicitly identified; excluded documents are never treated as current-product PASS evidence.
5. The 200-row thesis InsuranceQA subset is fully contained in the 1,000-row source and was not sampled independently.
"""


def gap_markdown(cases: Sequence[dict[str, Any]]) -> str:
    return f"""# Groundedness Annotation Gap Report

## Current state

- Candidate cases: **{len(cases)}**
- Reviewer 1 completed labels: **0**
- Reviewer 2 completed labels: **0**
- Adjudicated labels: **0**
- Final locked hold-out: **not defined or locked**
- Groundedness scores exposed to reviewers: **0**

## Blocking conditions

Final calibration requires both independent reviews, conflict adjudication,
complete adjudicated PASS/FAIL labels for eligible cases, explicit error
categories and criticality, and a finalized leakage-safe locked hold-out.

**No final threshold calibration was performed because independent human annotation and adjudication are still pending.**

## Next action

1. Give each reviewer only their dedicated workbook sheet.
2. Run `scripts/analyze_groundedness_annotations.py` after both reviews.
3. Adjudicate every conflict and finalize the hold-out by leakage group.
4. Run isolated scoring only after adjudication.
5. Run extended calibration only after all readiness checks pass.
"""


def main() -> int:
    insuranceqa = load_insuranceqa()
    active_paths = sorted(HELvetia_DIR.glob("*.pdf"))
    excluded_paths = sorted(EXCLUDED_PDF_DIR.glob("*.pdf"))
    active_records = extract_pdf_sentences(active_paths)
    excluded_records = extract_pdf_sentences(excluded_paths)

    cases = (
        build_insuranceqa_cases(insuranceqa)
        + build_helvetia_pdf_cases(active_records)
        + build_baloise_hard_negatives(active_records, excluded_records)
        + build_crm_cases()
        + build_legacy_cases()
    )
    if len(cases) != 498:
        raise AssertionError(f"Expected 498 candidates, got {len(cases)}")

    near_pairs = find_near_duplicates(cases)
    merge_leakage_groups(cases, near_pairs)
    split_plan = assign_group_splits(cases)
    quality = quality_checks(cases, near_pairs)

    expected = Counter(case["generator_expected_label"] for case in cases)
    if expected != Counter({"FAIL": 264, "PASS": 234}):
        raise AssertionError(f"Expected-label target drifted: {expected}")
    if quality["duplicate_case_ids"] or quality["invalid_source_cases"]:
        raise AssertionError("Blocking technical quality failure")
    if quality["leakage_group_split_violations"] or quality["cross_split_near_duplicate_pairs"]:
        raise AssertionError("Leakage-safe grouping failed")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SOURCE_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_JSONL, cases)
    write_review_csv(OUTPUT_REVIEW_CSV, cases)
    OUTPUT_SOURCE_AUDIT.write_text(
        source_audit_markdown(
            insuranceqa_rows=insuranceqa,
            active_pdf_records=active_records,
            excluded_pdf_records=excluded_records,
        ),
        encoding="utf-8",
    )
    OUTPUT_DICTIONARY.write_text(dictionary_markdown(), encoding="utf-8")
    OUTPUT_QUALITY.write_text(quality_markdown(cases, quality), encoding="utf-8")
    OUTPUT_LEAKAGE.write_text(
        leakage_markdown(cases, near_pairs, quality, split_plan), encoding="utf-8"
    )
    OUTPUT_GAP.write_text(gap_markdown(cases), encoding="utf-8")

    split_payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "random_seed": RANDOM_SEED,
        "status": "PRELIMINARY_PENDING_HUMAN_ANNOTATION",
        "final_locked_holdout": False,
        "assignment_unit": "leakage_group_id",
        "near_duplicate_method": "TF-IDF word unigram/bigram cosine similarity",
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "counts": {
            "total": len(cases),
            "generator_expected_label": dict(expected),
            "source_type": _count_rows(cases, "source_type"),
            "mutation_type": _count_rows(cases, "mutation_type"),
            "split": _count_rows(cases, "split"),
            "split_by_generator_expectation": {
                split: dict(Counter(
                    case["generator_expected_label"] for case in cases if case["split"] == split
                ))
                for split in sorted(set(case["split"] for case in cases))
            },
        },
        "group_plan": split_plan,
        "quality": quality,
        "limitations": [
            "Split balance uses generator expectations only and must be revisited after adjudication.",
            "A 27% hold-out within 498 candidates cannot contain both 100 PASS and 100 FAIL.",
            "No final hold-out may be opened or evaluated before threshold lock.",
        ],
        "case_assignments": [
            {
                "case_id": case["case_id"],
                "leakage_group_id": case["leakage_group_id"],
                "split": case["split"],
            }
            for case in cases
        ],
    }
    OUTPUT_SPLIT.write_text(
        json.dumps(split_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"found_sources=insuranceqa,helvetia_pdf,baloise_excluded_pdf,synthetic_crm,legacy_synthetic")
    print(f"insuranceqa_available={len(insuranceqa)}; used=190")
    print(f"helvetia_usable_sentence_records={len(active_records)}; used=100")
    print(f"baloise_excluded_sentence_records={len(excluded_records)}; used=20")
    print("crm_available=10 contacts,16 policies,8 claims; used=160")
    print("legacy_used=28")
    print(f"pass_candidates={expected['PASS']}")
    print(f"fail_candidates={expected['FAIL']}")
    print(f"hard_negatives={sum(bool(case['hard_negative']) for case in cases)}")
    print(f"source_type_counts={json.dumps(_count_rows(cases, 'source_type'), sort_keys=True)}")
    print(f"mutation_type_counts={json.dumps(_count_rows(cases, 'mutation_type'), sort_keys=True)}")
    print(f"duplicates={quality['exact_duplicate_triplets']}")
    print(f"excluded={quality['excluded_cases']}")
    print(f"split_counts={json.dumps(_count_rows(cases, 'split'), sort_keys=True)}")
    print(f"pending_annotation={len(cases)}")
    print("No final threshold calibration was performed because independent human annotation and adjudication are still pending.")
    for path in (
        OUTPUT_SOURCE_AUDIT,
        OUTPUT_JSONL,
        OUTPUT_REVIEW_CSV,
        OUTPUT_DICTIONARY,
        OUTPUT_QUALITY,
        OUTPUT_LEAKAGE,
        OUTPUT_SPLIT,
        OUTPUT_GAP,
    ):
        print(f"artifact={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
