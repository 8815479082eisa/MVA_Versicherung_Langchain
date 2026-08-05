from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.guardrails.integrations.nemo_actions import (  # noqa: E402
    GROUNDING_ALGORITHM_VERSION,
    calculate_groundedness_score,
)


DEFAULT_CANDIDATES = PROJECT_ROOT / "tests" / "fixtures" / "groundedness_extended_candidates.jsonl"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "reports" / "groundedness_dataset_review.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "groundedness_extended_scores.jsonl"
VALID_LABELS = {"PASS", "FAIL", "AMBIGUOUS", "EXCLUDE"}
VALID_CRITICALITIES = {"CRITICAL", "NON_CRITICAL", "REVIEW_REQUIRED"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_annotations(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".xlsx":
        from analyze_groundedness_annotations import read_rows

        return read_rows(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def readiness(candidates: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_ids = [str(row.get("case_id") or "") for row in candidates]
    annotation_ids = [str(row.get("case_id") or "") for row in annotations]
    if len(candidate_ids) != len(set(candidate_ids)) or len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Candidate and annotation case_id values must be unique")
    missing_annotations = sorted(set(candidate_ids) - set(annotation_ids))
    unknown_annotations = sorted(set(annotation_ids) - set(candidate_ids))
    by_id = {str(row["case_id"]): row for row in annotations}
    incomplete_review_1: list[str] = []
    incomplete_review_2: list[str] = []
    incomplete_adjudication: list[str] = []
    incomplete_reviewer_1_metadata: list[str] = []
    incomplete_reviewer_2_metadata: list[str] = []
    incomplete_adjudication_metadata: list[str] = []
    unresolved_conflicts: list[str] = []
    for case_id in candidate_ids:
        row = by_id.get(case_id, {})
        r1 = str(row.get("reviewer_1_label") or "").strip().upper()
        r2 = str(row.get("reviewer_2_label") or "").strip().upper()
        final = str(row.get("adjudicated_label") or "").strip().upper()
        if r1 not in VALID_LABELS:
            incomplete_review_1.append(case_id)
        if r2 not in VALID_LABELS:
            incomplete_review_2.append(case_id)
        if final not in VALID_LABELS:
            incomplete_adjudication.append(case_id)
        for prefix, target in (
            ("reviewer_1", incomplete_reviewer_1_metadata),
            ("reviewer_2", incomplete_reviewer_2_metadata),
        ):
            criticality = str(row.get(f"{prefix}_criticality") or "").strip().upper()
            category = str(row.get(f"{prefix}_error_category") or "").strip()
            notes = str(row.get(f"{prefix}_notes") or "").strip()
            confidence = str(row.get(f"{prefix}_confidence") or "").strip()
            if criticality not in VALID_CRITICALITIES or not category or not notes or confidence not in {"1", "2", "3", "4", "5"}:
                target.append(case_id)
        final_criticality = str(row.get("adjudicated_criticality") or "").strip().upper()
        final_category = str(row.get("adjudicated_error_category") or "").strip()
        final_confidence = str(row.get("adjudicated_confidence") or "").strip()
        if final_criticality not in VALID_CRITICALITIES or not final_category or final_confidence not in {"1", "2", "3", "4", "5"}:
            incomplete_adjudication_metadata.append(case_id)
        if r1 in VALID_LABELS and r2 in VALID_LABELS and r1 != r2:
            if final not in VALID_LABELS or not str(row.get("adjudication_notes") or "").strip():
                unresolved_conflicts.append(case_id)
    ready = not any(
        (missing_annotations, unknown_annotations, incomplete_review_1, incomplete_review_2,
         incomplete_adjudication, incomplete_reviewer_1_metadata,
         incomplete_reviewer_2_metadata, incomplete_adjudication_metadata,
         unresolved_conflicts)
    )
    return {
        "ready": ready,
        "candidate_count": len(candidates),
        "annotation_count": len(annotations),
        "missing_annotations": missing_annotations,
        "unknown_annotations": unknown_annotations,
        "incomplete_reviewer_1": incomplete_review_1,
        "incomplete_reviewer_2": incomplete_review_2,
        "incomplete_adjudication": incomplete_adjudication,
        "incomplete_reviewer_1_metadata": incomplete_reviewer_1_metadata,
        "incomplete_reviewer_2_metadata": incomplete_reviewer_2_metadata,
        "incomplete_adjudication_metadata": incomplete_adjudication_metadata,
        "unresolved_conflicts": unresolved_conflicts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score adjudicated extended groundedness cases without retrieval, LLM, CRM, or API calls"
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--readiness-only", action="store_true")
    args = parser.parse_args()

    candidates = _read_jsonl(args.candidates)
    annotations = _read_annotations(args.annotations)
    gate = readiness(candidates, annotations)
    gate_summary = {
        "ready": gate["ready"],
        "candidate_count": gate["candidate_count"],
        "annotation_count": gate["annotation_count"],
        "missing_annotations": len(gate["missing_annotations"]),
        "unknown_annotations": len(gate["unknown_annotations"]),
        "incomplete_reviewer_1": len(gate["incomplete_reviewer_1"]),
        "incomplete_reviewer_2": len(gate["incomplete_reviewer_2"]),
        "incomplete_adjudication": len(gate["incomplete_adjudication"]),
        "incomplete_reviewer_1_metadata": len(gate["incomplete_reviewer_1_metadata"]),
        "incomplete_reviewer_2_metadata": len(gate["incomplete_reviewer_2_metadata"]),
        "incomplete_adjudication_metadata": len(gate["incomplete_adjudication_metadata"]),
        "unresolved_conflicts": len(gate["unresolved_conflicts"]),
    }
    print(json.dumps(gate_summary, indent=2, ensure_ascii=False))
    if args.readiness_only:
        return 0 if gate["ready"] else 2
    if not gate["ready"]:
        raise SystemExit(
            "Refusing to calculate or reveal groundedness scores: two independent reviews "
            "and complete adjudication are required first."
        )

    annotation_by_id = {str(row["case_id"]): row for row in annotations}
    dataset_sha256 = hashlib.sha256(args.candidates.read_bytes()).hexdigest()
    scored: list[dict[str, Any]] = []
    for case in candidates:
        annotation = annotation_by_id[str(case["case_id"])]
        documents = [Document(page_content=str(text)) for text in case["context"]]
        score = calculate_groundedness_score(
            str(case["candidate_answer"]), documents, str(case["question"])
        )
        scored.append(
            {
                "case_id": case["case_id"],
                "split": case["split"],
                "leakage_group_id": case["leakage_group_id"],
                "source_type": case["source_type"],
                "product_group": case["product_group"],
                "mutation_type": case["mutation_type"],
                "adjudicated_label": str(annotation["adjudicated_label"]).strip().upper(),
                "adjudicated_error_category": str(annotation.get("adjudicated_error_category") or "").strip(),
                "adjudicated_criticality": str(annotation.get("adjudicated_criticality") or "").strip(),
                "groundedness_score": float(score),
                "algorithm_version": GROUNDING_ALGORITHM_VERSION,
                "candidate_dataset_sha256": dataset_sha256,
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"scored={len(scored)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
