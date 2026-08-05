from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_1 = ROOT / "reports" / "groundedness_human_review_reviewer_1_20260802.csv"
DEFAULT_REVIEW_2 = ROOT / "reports" / "groundedness_human_review_reviewer_2_20260802.csv"
DEFAULT_MAP = ROOT / "reports" / "groundedness_human_review_blind_map_20260802.json"
DEFAULT_COMBINED = ROOT / "reports" / "groundedness_human_review_combined.csv"
DEFAULT_CONFLICTS = ROOT / "reports" / "groundedness_human_review_conflicts.csv"
DEFAULT_SUMMARY = ROOT / "reports" / "groundedness_human_review_agreement.json"

ALLOWED_LABELS = {"SUPPORTED", "UNSUPPORTED", "AMBIGUOUS"}
ALLOWED_CRITICALITY = {"HIGH", "MEDIUM", "LOW"}
ALLOWED_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized(value: str) -> str:
    return (value or "").strip().upper()


def validate_review(rows: Sequence[dict[str, str]], name: str) -> dict[str, dict[str, str]]:
    if not rows:
        raise ValueError(f"{name} is empty")
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        blind_id = row.get("blind_case_id", "").strip()
        label = normalized(row.get("reviewer_label", ""))
        criticality = normalized(row.get("criticality", ""))
        confidence = normalized(row.get("confidence", ""))
        if not blind_id or blind_id in by_id:
            raise ValueError(f"{name}: missing or duplicate blind_case_id {blind_id!r}")
        if label not in ALLOWED_LABELS:
            raise ValueError(f"{name} {blind_id}: reviewer_label must be one of {sorted(ALLOWED_LABELS)}")
        if criticality not in ALLOWED_CRITICALITY:
            raise ValueError(f"{name} {blind_id}: criticality must be one of {sorted(ALLOWED_CRITICALITY)}")
        if confidence not in ALLOWED_CONFIDENCE:
            raise ValueError(f"{name} {blind_id}: confidence must be one of {sorted(ALLOWED_CONFIDENCE)}")
        row = dict(row)
        row["reviewer_label"] = label
        row["criticality"] = criticality
        row["confidence"] = confidence
        by_id[blind_id] = row
    return by_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge two completed blind groundedness reviews")
    parser.add_argument("--reviewer-1", type=Path, default=DEFAULT_REVIEW_1)
    parser.add_argument("--reviewer-2", type=Path, default=DEFAULT_REVIEW_2)
    parser.add_argument("--blind-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--combined", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--conflicts", type=Path, default=DEFAULT_CONFLICTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    for path in (args.combined, args.conflicts, args.summary):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    reviewer_1 = validate_review(read_csv(args.reviewer_1), "reviewer_1")
    reviewer_2 = validate_review(read_csv(args.reviewer_2), "reviewer_2")
    if set(reviewer_1) != set(reviewer_2):
        raise ValueError("Reviewer files do not contain the same blind_case_id set")
    mapping_payload = json.loads(args.blind_map.read_text(encoding="utf-8"))
    mapping = {row["blind_case_id"]: row for row in mapping_payload["mapping"]}
    if set(mapping) != set(reviewer_1):
        raise ValueError("Blind map does not match reviewer files")

    combined: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for blind_id in sorted(reviewer_1):
        first = reviewer_1[blind_id]
        second = reviewer_2[blind_id]
        labels_agree = first["reviewer_label"] == second["reviewer_label"]
        row = {
            "blind_case_id": blind_id,
            "case_id": mapping[blind_id]["case_id"],
            "split": mapping[blind_id]["split"],
            "leakage_group_id": mapping[blind_id]["leakage_group_id"],
            "reviewer_1_label": first["reviewer_label"],
            "reviewer_2_label": second["reviewer_label"],
            "labels_agree": labels_agree,
            "reviewer_1_error_category": first.get("error_category", ""),
            "reviewer_2_error_category": second.get("error_category", ""),
            "reviewer_1_criticality": first["criticality"],
            "reviewer_2_criticality": second["criticality"],
            "reviewer_1_confidence": first["confidence"],
            "reviewer_2_confidence": second["confidence"],
            "reviewer_1_notes": first.get("notes", ""),
            "reviewer_2_notes": second.get("notes", ""),
            "adjudicated_label": first["reviewer_label"] if labels_agree else "",
            "adjudicated_error_category": first.get("error_category", "") if labels_agree else "",
            "adjudicated_criticality": first["criticality"] if labels_agree and first["criticality"] == second["criticality"] else "",
            "adjudication_notes": "",
        }
        combined.append(row)
        if not labels_agree or first["reviewer_label"] == "AMBIGUOUS" or second["reviewer_label"] == "AMBIGUOUS":
            conflicts.append(row)

    write_csv(args.combined, combined)
    write_csv(args.conflicts, conflicts)
    agreements = sum(bool(row["labels_agree"]) for row in combined)
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(combined),
        "exact_label_agreements": agreements,
        "raw_agreement": agreements / len(combined),
        "conflict_or_ambiguous_count": len(conflicts),
        "reviewer_1_label_counts": dict(Counter(row["reviewer_1_label"] for row in combined)),
        "reviewer_2_label_counts": dict(Counter(row["reviewer_2_label"] for row in combined)),
        "warning": "No weak labels or groundedness scores were used to merge human reviews.",
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
