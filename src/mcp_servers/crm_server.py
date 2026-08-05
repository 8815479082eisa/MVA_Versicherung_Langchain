from __future__ import annotations

import hashlib
import logging
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.integrations.espocrm_client import CRMClientError, EspoCRMClient


logger = logging.getLogger(__name__)
_crm_client: EspoCRMClient | None = None


@lifespan
async def crm_lifespan(_: FastMCP):
    global _crm_client
    client = EspoCRMClient()
    _crm_client = client
    await client.start()
    try:
        yield {"crm_client": client}
    finally:
        await client.close()
        _crm_client = None


mcp = FastMCP(
    "Insurance CRM",
    instructions=(
        "Read-only access to structured customer, policy, and claim facts in "
        "EspoCRM. Never infer missing CRM records and never use these tools to "
        "answer general insurance coverage questions."
    ),
    lifespan=crm_lifespan,
    mask_error_details=True,
)


def _get_client() -> EspoCRMClient:
    if _crm_client is None:
        raise CRMClientError(
            "configuration",
            "The CRM MCP server has not completed startup.",
        )
    return _crm_client


def _result_count(payload: dict[str, Any]) -> int:
    if isinstance(payload.get("count"), int):
        return int(payload["count"])
    if payload.get("found") is True:
        return 1
    return 0


def _sanitized_reference(value: str | None) -> str:
    if not value:
        return "-"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


async def _execute_tool(
    tool_name: str,
    operation: Callable[[], Awaitable[dict[str, Any]]],
    *,
    entity_reference: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    success = False
    error_category = "-"
    count = 0
    try:
        payload = await operation()
        count = _result_count(payload)
        success = True
        return {
            "ok": True,
            "available": True,
            **payload,
        }
    except CRMClientError as exc:
        error_category = exc.category
        return {
            "ok": False,
            "available": exc.category not in {
                "disabled",
                "configuration",
                "authentication",
                "connection",
                "timeout",
            },
            "error": exc.as_dict(),
        }
    except Exception as exc:
        error_category = "internal"
        logger.error(
            "crm_tool unexpected_error tool=%s error_type=%s",
            tool_name,
            type(exc).__name__,
        )
        return {
            "ok": False,
            "available": False,
            "error": {
                "category": "internal",
                "message": "The CRM tool failed unexpectedly.",
            },
        }
    finally:
        logger.info(
            "crm_tool tool=%s success=%s duration_ms=%.1f result_count=%d "
            "entity_ref=%s error_category=%s",
            tool_name,
            success,
            (time.perf_counter() - started) * 1000,
            count,
            _sanitized_reference(entity_reference),
            error_category,
        )


async def _find_customer_impl(
    name: str | None = None,
    email: str | None = None,
) -> dict[str, Any]:
    return await _execute_tool(
        "find_customer",
        lambda: _get_client().find_customer(name=name, email=email),
    )


@mcp.tool()
async def find_customer(
    name: str | None = None,
    email: str | None = None,
) -> dict[str, Any]:
    """Find one customer by exact name or email before requesting their policies or claims.

    Use only for a user-authorized, specifically identified customer. Do not use
    this tool for broad customer enumeration. The response contains a minimal
    customer identifier and masks the email address.
    """

    return await _find_customer_impl(name=name, email=email)


async def _get_customer_policies_impl(customer_id: str) -> dict[str, Any]:
    return await _execute_tool(
        "get_customer_policies",
        lambda: _get_client().get_customer_policies(customer_id),
        entity_reference=customer_id,
    )


@mcp.tool()
async def get_customer_policies(customer_id: str) -> dict[str, Any]:
    """Return the identified customer's policies from CRM.

    Call find_customer first and pass its customer ID. Use this for policy
    numbers, product types, coverage types, status, dates, premiums, and
    deductibles. It does not explain what a coverage type generally covers.
    """

    return await _get_customer_policies_impl(customer_id)


async def _get_policy_impl(policy_number: str) -> dict[str, Any]:
    return await _execute_tool(
        "get_policy",
        lambda: _get_client().get_policy(policy_number),
        entity_reference=policy_number,
    )


@mcp.tool()
async def get_policy(policy_number: str) -> dict[str, Any]:
    """Return one policy by its exact policy number.

    Use this for structured policy facts stored in CRM. For general contractual
    coverage rules, use the document retrieval tool as well.
    """

    return await _get_policy_impl(policy_number)


async def _get_customer_claims_impl(customer_id: str) -> dict[str, Any]:
    return await _execute_tool(
        "get_customer_claims",
        lambda: _get_client().get_customer_claims(customer_id),
        entity_reference=customer_id,
    )


@mcp.tool()
async def get_customer_claims(customer_id: str) -> dict[str, Any]:
    """Return the identified customer's claims and their current CRM status.

    Call find_customer first. Use only for claims belonging to that customer;
    do not enumerate unrelated customers or expose raw CRM records.
    """

    return await _get_customer_claims_impl(customer_id)


async def _get_claim_status_impl(claim_number: str) -> dict[str, Any]:
    return await _execute_tool(
        "get_claim_status",
        lambda: _get_client().get_claim_status(claim_number),
        entity_reference=claim_number,
    )


@mcp.tool()
async def get_claim_status(claim_number: str) -> dict[str, Any]:
    """Return the current CRM status and minimal facts for one exact claim number.

    This tool reports operational claim status only. It must not be used to
    decide whether a loss is contractually covered without document evidence.
    """

    return await _get_claim_status_impl(claim_number)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    mcp.run(transport="stdio", show_banner=False)
