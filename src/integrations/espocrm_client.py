from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Literal, Sequence
from urllib.parse import quote

import httpx

from src.config.models import bootstrap_environment


logger = logging.getLogger(__name__)

CRMErrorCategory = Literal[
    "disabled",
    "configuration",
    "authentication",
    "connection",
    "timeout",
    "not_found",
    "duplicate_match",
    "malformed_response",
    "upstream",
    "invalid_request",
]

_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class CRMClientError(RuntimeError):
    def __init__(
        self,
        category: CRMErrorCategory,
        message: str,
        *,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.category = category
        self.public_message = message
        self.status_code = status_code

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "category": self.category,
            "message": self.public_message,
        }
        if self.status_code is not None:
            payload["status_code"] = self.status_code
        return payload


@dataclass(frozen=True)
class EspoCRMConfig:
    enabled: bool
    base_url: str
    api_key: str
    request_timeout_seconds: float = 3.0
    max_pages: int = 10
    page_size: int = 50

    @classmethod
    def from_environment(cls) -> "EspoCRMConfig":
        bootstrap_environment()

        timeout = _bounded_float(
            os.getenv("ESPOCRM_REQUEST_TIMEOUT_SECONDS"),
            default=3.0,
            minimum=0.2,
            maximum=30.0,
        )
        max_pages = _bounded_int(
            os.getenv("ESPOCRM_MAX_PAGES"),
            default=10,
            minimum=1,
            maximum=100,
        )
        page_size = _bounded_int(
            os.getenv("ESPOCRM_PAGE_SIZE"),
            default=50,
            minimum=1,
            maximum=200,
        )

        return cls(
            enabled=_env_flag("CRM_ENABLED", default=False),
            base_url=os.getenv("ESPOCRM_BASE_URL", "http://espocrm").strip(),
            api_key=os.getenv("ESPOCRM_API_KEY", "").strip(),
            request_timeout_seconds=timeout,
            max_pages=max_pages,
            page_size=page_size,
        )

    @property
    def api_base_url(self) -> str:
        value = self.base_url.rstrip("/")
        return value if value.endswith("/api/v1") else f"{value}/api/v1"

    def availability_error(self) -> CRMClientError | None:
        if not self.enabled:
            return CRMClientError(
                "disabled",
                "CRM integration is disabled. Set CRM_ENABLED=true to enable it.",
            )
        if not self.base_url:
            return CRMClientError(
                "configuration",
                "CRM integration is enabled but ESPOCRM_BASE_URL is missing.",
            )
        if not self.api_key:
            return CRMClientError(
                "configuration",
                "CRM integration is enabled but ESPOCRM_API_KEY is missing.",
            )
        return None


class EspoCRMClient:
    """Small read-only EspoCRM REST client with normalized insurance responses."""

    def __init__(
        self,
        config: EspoCRMConfig | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config = config or EspoCRMConfig.from_environment()
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._lifecycle_lock = asyncio.Lock()

    @property
    def started(self) -> bool:
        return self._client is not None

    async def start(self) -> None:
        if self._client is not None:
            return

        async with self._lifecycle_lock:
            if self._client is not None:
                return

            availability_error = self.config.availability_error()
            if availability_error is not None:
                return

            timeout = httpx.Timeout(
                connect=self.config.request_timeout_seconds,
                read=self.config.request_timeout_seconds,
                write=self.config.request_timeout_seconds,
                pool=self.config.request_timeout_seconds,
            )
            limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
            self._client = httpx.AsyncClient(
                base_url=self.config.api_base_url,
                headers={
                    "X-Api-Key": self.config.api_key,
                    "Accept": "application/json",
                },
                timeout=timeout,
                limits=limits,
                transport=self._transport,
            )

    async def close(self) -> None:
        async with self._lifecycle_lock:
            client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def __aenter__(self) -> "EspoCRMClient":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def _require_available(self) -> httpx.AsyncClient:
        availability_error = self.config.availability_error()
        if availability_error is not None:
            raise availability_error
        if self._client is None:
            raise CRMClientError(
                "configuration",
                "The CRM HTTP client has not been started.",
            )
        return self._client

    async def _request_json(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        client = self._require_available()
        request_started = time.perf_counter()
        last_error: CRMClientError | None = None

        for attempt in range(2):
            try:
                response = await client.get(endpoint, params=params)
            except httpx.TimeoutException:
                last_error = CRMClientError(
                    "timeout",
                    "EspoCRM did not respond before the configured timeout.",
                )
                if attempt == 0:
                    continue
                raise last_error
            except (httpx.ConnectError, httpx.NetworkError):
                last_error = CRMClientError(
                    "connection",
                    "EspoCRM is unavailable or the connection failed.",
                )
                if attempt == 0:
                    continue
                raise last_error
            except httpx.RequestError as exc:
                raise CRMClientError(
                    "connection",
                    "The EspoCRM request could not be completed.",
                ) from exc

            if response.status_code in {401, 403}:
                raise CRMClientError(
                    "authentication",
                    "The EspoCRM API user is not authenticated or lacks read permission.",
                    status_code=response.status_code,
                )
            if response.status_code == 404:
                raise CRMClientError(
                    "not_found",
                    "The requested EspoCRM resource was not found.",
                    status_code=404,
                )
            if response.status_code in _RETRYABLE_STATUS_CODES and attempt == 0:
                retry_after = response.headers.get("Retry-After", "")
                try:
                    delay = min(max(float(retry_after), 0.0), 0.25)
                except ValueError:
                    delay = 0.05
                if delay:
                    await asyncio.sleep(delay)
                continue
            if response.status_code >= 400:
                raise CRMClientError(
                    "upstream",
                    "EspoCRM returned an unexpected error.",
                    status_code=response.status_code,
                )

            try:
                payload = response.json()
            except (ValueError, json.JSONDecodeError) as exc:
                raise CRMClientError(
                    "malformed_response",
                    "EspoCRM returned a malformed JSON response.",
                ) from exc
            if not isinstance(payload, dict):
                raise CRMClientError(
                    "malformed_response",
                    "EspoCRM returned an unexpected response structure.",
                )

            logger.debug(
                "crm_rest success=true duration_ms=%.1f",
                (time.perf_counter() - request_started) * 1000,
            )
            return payload

        raise last_error or CRMClientError(
            "upstream",
            "EspoCRM did not return a usable response.",
        )

    async def _list_records(
        self,
        entity_or_relation: str,
        *,
        fields: Sequence[str],
        where: Sequence[dict[str, Any]] | None = None,
        maximum_records: int = 100,
        order_by: str | None = None,
        order: str = "asc",
    ) -> list[dict[str, Any]]:
        maximum_records = max(1, min(maximum_records, 500))
        records: list[dict[str, Any]] = []
        offset = 0

        for _ in range(self.config.max_pages):
            remaining = maximum_records - len(records)
            if remaining <= 0:
                break

            page_size = min(self.config.page_size, remaining)
            search_params: dict[str, Any] = {
                "select": list(fields),
                "offset": offset,
                "maxSize": page_size,
            }
            if where:
                search_params["where"] = list(where)
            if order_by:
                search_params["orderBy"] = order_by
                search_params["order"] = order

            payload = await self._request_json(
                entity_or_relation,
                params={
                    "searchParams": json.dumps(
                        search_params,
                        separators=(",", ":"),
                    )
                },
            )
            page = payload.get("list")
            if not isinstance(page, list) or any(
                not isinstance(item, dict) for item in page
            ):
                raise CRMClientError(
                    "malformed_response",
                    "EspoCRM returned a malformed record collection.",
                )

            records.extend(page)
            offset += len(page)
            total = payload.get("total")
            if not page or (isinstance(total, int) and offset >= total):
                break
            if len(page) < page_size:
                break

        return records

    async def health(self) -> dict[str, Any]:
        await self._request_json("App/user")
        return {"available": True, "authenticated": True}

    async def find_customer(
        self,
        *,
        name: str | None = None,
        email: str | None = None,
    ) -> dict[str, Any]:
        clean_name = (name or "").strip()
        clean_email = (email or "").strip().lower()
        if not clean_name and not clean_email:
            raise CRMClientError(
                "invalid_request",
                "Provide a customer name or email address.",
            )

        where: list[dict[str, Any]] = []
        if clean_email:
            where.append(
                {
                    "type": "equals",
                    "attribute": "emailAddress",
                    "value": clean_email,
                }
            )
        if clean_name:
            name_parts = clean_name.split()
            if len(name_parts) >= 2:
                where.extend(
                    [
                        {
                            "type": "equals",
                            "attribute": "firstName",
                            "value": name_parts[0],
                        },
                        {
                            "type": "equals",
                            "attribute": "lastName",
                            "value": " ".join(name_parts[1:]),
                        },
                    ]
                )
            else:
                where.append(
                    {
                        "type": "contains",
                        "attribute": "name",
                        "value": clean_name,
                    }
                )

        records = await self._list_records(
            "Contact",
            fields=(
                "id",
                "firstName",
                "lastName",
                "emailAddress",
                "phoneNumber",
                "addressStreet",
                "addressCity",
                "addressPostalCode",
                "addressState",
                "addressCountry",
                "birthDate",
            ),
            where=where,
            maximum_records=2,
        )
        if not records:
            return {"found": False, "customer": None}
        if len(records) > 1:
            return {
                "found": False,
                "ambiguous": True,
                "match_count": len(records),
                "customer": None,
            }

        record = records[0]
        first_name = _clean_text(record.get("firstName"))
        last_name = _clean_text(record.get("lastName"))
        display_name = " ".join(part for part in (first_name, last_name) if part)
        return {
            "found": True,
            "customer": {
                "id": _required_text(record, "id"),
                "display_name": display_name or "Unnamed customer",
                "email": _clean_text(record.get("emailAddress")),
                "phone": _clean_text(record.get("phoneNumber")),
                "address": _format_contact_address(record),
                "date_of_birth": _clean_text(record.get("birthDate")),
            },
        }

    async def get_customer_policies(self, customer_id: str) -> dict[str, Any]:
        safe_id = _validate_identifier(customer_id, label="customer_id")
        records = await self._list_records(
            f"Contact/{quote(safe_id, safe='')}/mvaPolicies",
            fields=(
                "id",
                "name",
                "policyNumber",
                "productType",
                "coverageType",
                "status",
                "startDate",
                "endDate",
                "deductible",
                "annualPremium",
                "currency",
                "customerId",
                "customerName",
            ),
            maximum_records=100,
            order_by="startDate",
            order="desc",
        )
        return {
            "customer_id": safe_id,
            "count": len(records),
            "policies": [_normalize_policy(record) for record in records],
        }

    async def get_policy(self, policy_number: str) -> dict[str, Any]:
        clean_number = _validate_business_number(
            policy_number,
            label="policy_number",
        )
        records = await self._list_records(
            "MvaPolicy",
            fields=(
                "id",
                "name",
                "policyNumber",
                "productType",
                "coverageType",
                "status",
                "startDate",
                "endDate",
                "deductible",
                "annualPremium",
                "currency",
                "customerId",
                "customerName",
            ),
            where=(
                {
                    "type": "equals",
                    "attribute": "policyNumber",
                    "value": clean_number,
                },
            ),
            maximum_records=2,
        )
        if not records:
            return {"found": False, "policy": None}
        if len(records) > 1:
            raise CRMClientError(
                "duplicate_match",
                "Multiple policies match the supplied policy number.",
            )
        return {"found": True, "policy": _normalize_policy(records[0])}

    async def get_customer_claims(self, customer_id: str) -> dict[str, Any]:
        safe_id = _validate_identifier(customer_id, label="customer_id")
        records = await self._list_records(
            f"Contact/{quote(safe_id, safe='')}/mvaClaims",
            fields=(
                "id",
                "name",
                "claimNumber",
                "claimDate",
                "damageType",
                "claimedAmount",
                "currency",
                "status",
                "policyId",
                "policyName",
                "customerId",
                "customerName",
                "description",
            ),
            maximum_records=100,
            order_by="claimDate",
            order="desc",
        )
        return {
            "customer_id": safe_id,
            "count": len(records),
            "claims": [_normalize_claim(record) for record in records],
        }

    async def get_claim_status(self, claim_number: str) -> dict[str, Any]:
        clean_number = _validate_business_number(
            claim_number,
            label="claim_number",
        )
        records = await self._list_records(
            "MvaClaim",
            fields=(
                "id",
                "name",
                "claimNumber",
                "claimDate",
                "damageType",
                "claimedAmount",
                "currency",
                "status",
                "policyId",
                "policyName",
                "customerId",
                "customerName",
                "description",
            ),
            where=(
                {
                    "type": "equals",
                    "attribute": "claimNumber",
                    "value": clean_number,
                },
            ),
            maximum_records=2,
        )
        if not records:
            return {"found": False, "claim": None}
        if len(records) > 1:
            raise CRMClientError(
                "duplicate_match",
                "Multiple claims match the supplied claim number.",
            )
        return {"found": True, "claim": _normalize_claim(records[0])}


def _normalize_policy(record: dict[str, Any]) -> dict[str, Any]:
    currency = _clean_text(record.get("currency")) or "EUR"
    return {
        "id": _required_text(record, "id"),
        "name": _clean_text(record.get("name")),
        "policy_number": _clean_text(record.get("policyNumber")),
        "product_type": _clean_text(record.get("productType")),
        "coverage_type": _clean_text(record.get("coverageType")),
        "status": _clean_text(record.get("status")),
        "deductible": {
            "amount": _number(record.get("deductible")),
            "currency": currency,
        },
        "annual_premium": {
            "amount": _number(record.get("annualPremium")),
            "currency": currency,
        },
        "start_date": _clean_text(record.get("startDate")),
        "end_date": _clean_text(record.get("endDate")),
        "customer_id": _clean_text(record.get("customerId")),
        "customer_name": _clean_text(record.get("customerName")),
    }


def _normalize_claim(record: dict[str, Any]) -> dict[str, Any]:
    currency = _clean_text(record.get("currency")) or "EUR"
    return {
        "id": _required_text(record, "id"),
        "name": _clean_text(record.get("name")),
        "claim_number": _clean_text(record.get("claimNumber")),
        "claim_date": _clean_text(record.get("claimDate")),
        "damage_type": _clean_text(record.get("damageType")),
        "claimed_amount": {
            "amount": _number(record.get("claimedAmount")),
            "currency": currency,
        },
        "status": _clean_text(record.get("status")),
        "policy_reference": _clean_text(record.get("policyName")),
        "policy_id": _clean_text(record.get("policyId")),
        "customer_id": _clean_text(record.get("customerId")),
        "customer_name": _clean_text(record.get("customerName")),
        "description": _clean_text(record.get("description")),
    }


def _format_contact_address(record: dict[str, Any]) -> str:
    locality = " ".join(
        part
        for part in (
            _clean_text(record.get("addressPostalCode")),
            _clean_text(record.get("addressCity")),
        )
        if part
    )
    parts = [
        _clean_text(record.get("addressStreet")),
        locality,
        _clean_text(record.get("addressState")),
        _clean_text(record.get("addressCountry")),
    ]
    return ", ".join(part for part in parts if part)


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _bounded_float(
    raw: str | None,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(raw) if raw is not None else default
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def _bounded_int(
    raw: str | None,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(raw) if raw is not None else default
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _required_text(record: dict[str, Any], field: str) -> str:
    value = _clean_text(record.get(field))
    if not value:
        raise CRMClientError(
            "malformed_response",
            f"EspoCRM omitted required field '{field}' from a record.",
        )
    return value


def _number(value: Any) -> int | float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CRMClientError(
            "malformed_response",
            "EspoCRM returned a non-numeric insurance amount.",
        ) from exc
    return int(number) if number.is_integer() else number


def _mask_email(value: str) -> str | None:
    if not value or "@" not in value:
        return None
    local, domain = value.split("@", 1)
    return f"{local[:1]}***@{domain}"


def _validate_identifier(value: str, *, label: str) -> str:
    clean_value = (value or "").strip()
    if not _SAFE_IDENTIFIER.fullmatch(clean_value):
        raise CRMClientError(
            "invalid_request",
            f"{label} has an invalid format.",
        )
    return clean_value


def _validate_business_number(value: str, *, label: str) -> str:
    clean_value = (value or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9_-]{2,63}", clean_value):
        raise CRMClientError(
            "invalid_request",
            f"{label} has an invalid format.",
        )
    return clean_value
