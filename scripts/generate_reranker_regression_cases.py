from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
POLICIES_PATH = ROOT / "data" / "synthetic" / "crm" / "policies.csv"
CLAIMS_PATH = ROOT / "data" / "synthetic" / "crm" / "claims.csv"
OUTPUT_PATH = ROOT / "tests" / "fixtures" / "reranker_regression_cases.json"
CANDIDATE_COUNT = 8

PRODUCT_DE = {
    "Motor Insurance": "Kfz-Versicherung",
    "Helvetia Motor Vehicle Insurance": "Helvetia Kfz-Versicherung",
    "Helvetia Household Contents Insurance": "Helvetia Hausratversicherung",
    "Helvetia Private Liability Insurance": "Helvetia Privathaftpflichtversicherung",
    "Helvetia Legal Protection Insurance": "Helvetia Rechtsschutzversicherung",
    "Personal Liability": "Privathaftpflichtversicherung",
    "Household Insurance": "Hausratversicherung",
    "Legal Protection": "Rechtsschutzversicherung",
}
STATUS_DE = {
    "Active": "aktiv",
    "Pending": "ausstehend",
    "Expired": "abgelaufen",
    "Cancelled": "gekündigt",
    "Under Review": "in Prüfung",
    "Approved": "genehmigt",
    "Paid": "bezahlt",
    "Open": "offen",
    "Rejected": "abgelehnt",
    "Closed": "geschlossen",
}
DAMAGE_DE = {
    "Glass Damage": "Glasschaden",
    "Parking Damage": "Parkschaden",
    "Water Damage": "Wasserschaden",
    "Liability Damage": "Haftpflichtschaden",
    "Personal Liability": "Privathaftpflichtschaden",
    "Theft": "Diebstahl",
    "Collision": "Kollisionsschaden",
    "Property Damage": "Sachschaden",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def customer_name(email: str) -> str:
    local = email.split("@", 1)[0]
    return " ".join(part.capitalize() for part in local.split("."))


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def policy_document(row: dict[str, str]) -> str:
    name = customer_name(row["customerEmail"])
    return (
        "Helvetia customer policy schedule. This record contains individual contract data "
        "and takes precedence over generic product descriptions when a question asks for "
        f"customer-specific values. Customer: {name}. Policy number: {row['policyNumber']}. "
        f"Product: {row['productType']}. Coverage type: {row['coverageType']}. "
        f"Status: {row['status']}. Contract term: {row['startDate']} through {row['endDate']}. "
        f"Individual deductible: {row['deductible']} {row['currency']}. Annual premium: "
        f"{row['annualPremium']} {row['currency']}. The deductible and premium in this "
        "schedule apply only to the named policy and must not be transferred to another "
        "customer, product, earlier contract, or later contract. General insurance terms "
        "may explain insured events, exclusions, notification duties and settlement rules, "
        "but they do not replace these customer-specific monetary values or effective dates."
    )


def claim_document(row: dict[str, str]) -> str:
    name = customer_name(row["customerEmail"])
    return (
        "Helvetia customer claim record. This is an individual claim entry and must be "
        "distinguished from similar claims, generic coverage terms and the associated policy "
        f"schedule. Customer: {name}. Claim number: {row['claimNumber']}. Claim date: "
        f"{row['claimDate']}. Damage type: {row['damageType']}. Description: "
        f"{row['description']}. Claimed amount: {row['claimedAmount']} {row['currency']}. "
        f"Claim status: {row['status']}. Referenced policy: {row['policyNumber']}. The status "
        "describes this claim only; it does not state whether another customer's claim was "
        "approved or whether every event of the same damage type is covered. The amount is "
        "the submitted claim value and is not an annual premium or policy deductible."
    )


def prioritized_rows(
    correct: dict[str, str],
    rows: Iterable[dict[str, str]],
    *,
    kind: str,
) -> list[dict[str, str]]:
    remaining = [row for row in rows if row is not correct]
    if kind == "policy":
        keys = (
            lambda row: row["customerEmail"] == correct["customerEmail"],
            lambda row: row["productType"] == correct["productType"],
            lambda row: row["coverageType"] == correct["coverageType"],
            lambda row: row["status"] == correct["status"],
        )
    else:
        keys = (
            lambda row: row["customerEmail"] == correct["customerEmail"],
            lambda row: row["damageType"] == correct["damageType"],
            lambda row: row["status"] == correct["status"],
            lambda row: row["policyNumber"] == correct["policyNumber"],
        )
    chosen: list[dict[str, str]] = []
    for predicate in keys:
        for row in remaining:
            if predicate(row) and row not in chosen:
                chosen.append(row)
                if len(chosen) == CANDIDATE_COUNT - 1:
                    return chosen
    for row in remaining:
        if row not in chosen:
            chosen.append(row)
            if len(chosen) == CANDIDATE_COUNT - 1:
                break
    return chosen


def policy_query(row: dict[str, str], language: str) -> str:
    name = customer_name(row["customerEmail"])
    product_en = row["productType"]
    product_de = PRODUCT_DE[product_en]
    status_de = STATUS_DE[row["status"]]
    qualifier_en = f"with status {row['status']} starting on {row['startDate']}"
    qualifier_de = f"mit Status {status_de} und Beginn am {row['startDate']}"
    if row["policyNumber"] == "TEST-KFZ-2026-1003":
        qualifier_en = "that is current and newest as of 2026-08-01"
        qualifier_de = "die am 01.08.2026 aktuell und die neueste ist"
    elif row["policyNumber"] == "TEST-KFZ-2026-1001":
        qualifier_en = "that is the earlier contract starting on 2026-01-01"
        qualifier_de = "die der ältere Vertrag mit Beginn am 01.01.2026 ist"

    if language == "en":
        return (
            f"For {name}, identify the {product_en} policy {qualifier_en}. What are its "
            "policy number, coverage type, deductible, annual premium, status and full term?"
        )
    if language == "de":
        return (
            f"Identifiziere für {name} die {product_de}, {qualifier_de}. Wie lauten "
            "Policennummer, Deckungsart, Selbstbeteiligung, Jahresprämie, Status und Laufzeit?"
        )
    return (
        f"Welche {product_de} von {name} ist {qualifier_de}? Return policy number, coverage "
        "type, deductible, annual premium, status and contract term."
    )


def claim_query(row: dict[str, str], language: str) -> str:
    name = customer_name(row["customerEmail"])
    if language == "en":
        return (
            f"Find {name}'s {row['damageType']} claim dated {row['claimDate']}. What are the "
            "claim number, claimed amount, status and referenced policy number?"
        )
    return (
        f"Finde den {DAMAGE_DE[row['damageType']]} von {name} vom {row['claimDate']}. "
        "Wie lauten Schadennummer, gemeldeter Betrag, Bearbeitungsstatus und zugehörige "
        "Policennummer?"
    )


def make_candidates(
    correct: dict[str, str],
    rows: list[dict[str, str]],
    *,
    kind: str,
) -> list[dict[str, Any]]:
    id_field = "policyNumber" if kind == "policy" else "claimNumber"
    renderer = policy_document if kind == "policy" else claim_document
    selected = [correct, *prioritized_rows(correct, rows, kind=kind)]
    return [
        {
            "id": f"{kind}_{slug(row[id_field])}",
            "relevance": 3 if row is correct else 0,
            "label": "exact answer-bearing record" if row is correct else "hard negative",
            "document": renderer(row),
        }
        for row in selected
    ]


def build_cases() -> dict[str, Any]:
    policies = read_rows(POLICIES_PATH)
    claims = read_rows(CLAIMS_PATH)
    cases: list[dict[str, Any]] = []
    for row in policies:
        for language in ("en", "de", "mixed"):
            cases.append(
                {
                    "id": f"policy_{slug(row['policyNumber'])}_{language}",
                    "language": language,
                    "mandatory_rank1": row["policyNumber"] == "TEST-KFZ-2026-1003",
                    "query": policy_query(row, language),
                    "correct_candidate_id": f"policy_{slug(row['policyNumber'])}",
                    "candidates": make_candidates(row, policies, kind="policy"),
                }
            )
    for row in claims:
        for language in ("en", "de"):
            cases.append(
                {
                    "id": f"claim_{slug(row['claimNumber'])}_{language}",
                    "language": language,
                    "mandatory_rank1": False,
                    "query": claim_query(row, language),
                    "correct_candidate_id": f"claim_{slug(row['claimNumber'])}",
                    "candidates": make_candidates(row, claims, kind="claim"),
                }
            )
    return {
        "schema_version": 1,
        "language": "en,de,mixed",
        "candidate_count_per_query": CANDIDATE_COUNT,
        "relevance_scale": {
            "0": "irrelevant or hard negative",
            "3": "exact answer-bearing record",
        },
        "provenance": [
            str(POLICIES_PATH.relative_to(ROOT)).replace("\\", "/"),
            str(CLAIMS_PATH.relative_to(ROOT)).replace("\\", "/"),
        ],
        "generation": {
            "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
            "policy_cases": len(policies) * 3,
            "claim_cases": len(claims) * 2,
        },
        "cases": cases,
    }


def main() -> int:
    payload = build_cases()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(payload['cases'])} cases to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
