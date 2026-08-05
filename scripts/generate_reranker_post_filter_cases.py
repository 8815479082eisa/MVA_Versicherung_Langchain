from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "tests" / "fixtures" / "reranker_regression_cases.json"
OUTPUT_PATH = ROOT / "tests" / "fixtures" / "reranker_post_filter_cases.json"
CRM_DIR = ROOT / "data" / "synthetic" / "crm"
REFERENCE_DATE = "2026-08-01"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def candidate_id(kind: str, identifier: str) -> str:
    return f"{kind}_{identifier.lower().replace('-', '_')}"


def main() -> int:
    payload: dict[str, Any] = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    contacts = {
        row["emailAddress"]: f"{row['firstName']} {row['lastName']}"
        for row in read_rows(CRM_DIR / "contacts.csv")
    }
    metadata: dict[str, dict[str, Any]] = {}
    for row in read_rows(CRM_DIR / "policies.csv"):
        metadata[candidate_id("policy", row["policyNumber"])] = {
            "entity_type": "policy",
            "policy_number": row["policyNumber"],
            "customer_name": contacts[row["customerEmail"]],
            "product_type": row["productType"],
            "coverage_type": row["coverageType"],
            "status": row["status"],
            "start_date": row["startDate"],
            "end_date": row["endDate"],
        }
    for row in read_rows(CRM_DIR / "claims.csv"):
        metadata[candidate_id("claim", row["claimNumber"])] = {
            "entity_type": "claim",
            "claim_number": row["claimNumber"],
            "customer_name": contacts[row["customerEmail"]],
            "status": row["status"],
            "claim_date": row["claimDate"],
            "policy_number": row["policyNumber"],
        }

    for case in payload["cases"]:
        for candidate in case["candidates"]:
            try:
                candidate["metadata"] = metadata[candidate["id"]]
            except KeyError as exc:
                raise ValueError(f"No CRM metadata for {candidate['id']}") from exc
    payload["schema_version"] = 2
    payload["selection_reference_date"] = REFERENCE_DATE
    payload["provenance"] = [
        *payload.get("provenance", []),
        "data/synthetic/crm/contacts.csv",
    ]
    payload["generation"] = {
        **payload.get("generation", {}),
        "post_filter_script": "scripts/generate_reranker_post_filter_cases.py",
        "source_fixture": "tests/fixtures/reranker_regression_cases.json",
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(payload['cases'])} cases to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
