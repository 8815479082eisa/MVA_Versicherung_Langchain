from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.models import bootstrap_environment


DATA_ROOT = PROJECT_ROOT / "data" / "synthetic" / "crm"


class SeedError(RuntimeError):
    pass


@dataclass
class SeedStats:
    created: int = 0
    skipped: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"created": self.created, "skipped": self.skipped}


class EspoCRMSeeder:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: float = 5.0):
        self.base_url = base_url.rstrip("/") + "/api/v1"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "X-Api-Key": api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(
                connect=timeout_seconds,
                read=timeout_seconds,
                write=timeout_seconds,
                pool=timeout_seconds,
            ),
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self.client.request(
            method,
            endpoint,
            params=params,
            json=json_body,
        )
        if response.status_code in {401, 403}:
            raise SeedError(
                "The one-time seed API user is not authenticated or lacks create permission."
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise SeedError(f"Malformed EspoCRM response for {method} {endpoint}.")
        return payload

    async def check_authentication(self) -> None:
        await self._request("GET", "App/user")

    async def find_unique(
        self,
        entity: str,
        attribute: str,
        value: str,
    ) -> dict[str, Any] | None:
        search_params = {
            "select": ["id", attribute],
            "maxSize": 2,
            "where": [
                {
                    "type": "equals",
                    "attribute": attribute,
                    "value": value,
                }
            ],
        }
        payload = await self._request(
            "GET",
            entity,
            params={"searchParams": json.dumps(search_params, separators=(",", ":"))},
        )
        records = payload.get("list")
        if not isinstance(records, list):
            raise SeedError(f"Malformed record list returned for {entity}.")
        if len(records) > 1:
            raise SeedError(
                f"Duplicate {entity} records found for the configured synthetic natural key."
            )
        return records[0] if records else None

    async def ensure_record(
        self,
        entity: str,
        natural_key: str,
        natural_value: str,
        body: dict[str, Any],
        stats: SeedStats,
    ) -> dict[str, Any]:
        existing = await self.find_unique(entity, natural_key, natural_value)
        if existing is not None:
            stats.skipped += 1
            return existing

        created = await self._request("POST", entity, json_body=body)
        if not isinstance(created.get("id"), str):
            raise SeedError(f"EspoCRM did not return an ID after creating {entity}.")
        stats.created += 1
        return created


def _read_csv(name: str) -> list[dict[str, str]]:
    path = DATA_ROOT / name
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


async def seed() -> dict[str, Any]:
    bootstrap_environment()
    base_url = (
        os.getenv("ESPOCRM_SEED_BASE_URL")
        or os.getenv("ESPOCRM_PUBLIC_URL")
        or "http://localhost:8080"
    ).strip()
    api_key = os.getenv("ESPOCRM_SEED_API_KEY", "").strip()
    if not api_key:
        raise SeedError(
            "ESPOCRM_SEED_API_KEY is required. Use a temporary API user with create "
            "permission only for Contact, Policy, and Claim, then revoke it."
        )

    seeder = EspoCRMSeeder(base_url, api_key)
    contact_stats = SeedStats()
    policy_stats = SeedStats()
    claim_stats = SeedStats()
    contacts_by_email: dict[str, str] = {}
    policies_by_number: dict[str, str] = {}

    try:
        await seeder.check_authentication()

        for row in _read_csv("contacts.csv"):
            record = await seeder.ensure_record(
                "Contact",
                "emailAddress",
                row["emailAddress"],
                {
                    "firstName": row["firstName"],
                    "lastName": row["lastName"],
                    "emailAddress": row["emailAddress"],
                },
                contact_stats,
            )
            contacts_by_email[row["emailAddress"]] = str(record["id"])

        for row in _read_csv("policies.csv"):
            customer_id = contacts_by_email[row["customerEmail"]]
            record = await seeder.ensure_record(
                "MvaPolicy",
                "policyNumber",
                row["policyNumber"],
                {
                    "name": row["name"],
                    "policyNumber": row["policyNumber"],
                    "productType": row["productType"],
                    "coverageType": row["coverageType"],
                    "status": row["status"],
                    "startDate": row["startDate"],
                    "endDate": row["endDate"],
                    "deductible": float(row["deductible"]),
                    "annualPremium": float(row["annualPremium"]),
                    "currency": row["currency"],
                    "customerId": customer_id,
                },
                policy_stats,
            )
            policies_by_number[row["policyNumber"]] = str(record["id"])

        for row in _read_csv("claims.csv"):
            await seeder.ensure_record(
                "MvaClaim",
                "claimNumber",
                row["claimNumber"],
                {
                    "name": row["name"],
                    "claimNumber": row["claimNumber"],
                    "claimDate": row["claimDate"],
                    "damageType": row["damageType"],
                    "description": row["description"],
                    "claimedAmount": float(row["claimedAmount"]),
                    "currency": row["currency"],
                    "status": row["status"],
                    "customerId": contacts_by_email[row["customerEmail"]],
                    "policyId": policies_by_number[row["policyNumber"]],
                },
                claim_stats,
            )
    finally:
        await seeder.close()

    return {
        "synthetic_data_only": True,
        "contacts": contact_stats.as_dict(),
        "policies": policy_stats.as_dict(),
        "claims": claim_stats.as_dict(),
    }


def main() -> int:
    try:
        result = asyncio.run(seed())
    except (SeedError, httpx.HTTPError, KeyError, ValueError) as exc:
        print(f"SEED_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
