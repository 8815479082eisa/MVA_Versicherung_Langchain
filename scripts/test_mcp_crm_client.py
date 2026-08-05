from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mcp_clients.insurance_tools import PersistentMCPToolRuntime


EXPECTED_TOOLS = {
    "find_customer",
    "get_customer_policies",
    "get_policy",
    "get_customer_claims",
    "get_claim_status",
}


async def main() -> None:
    runtime = PersistentMCPToolRuntime()
    try:
        await runtime.start(("insurance_crm",))
        discovered_tools = set(runtime.tool_names)
        if discovered_tools != EXPECTED_TOOLS:
            raise RuntimeError(
                "Unexpected CRM MCP tool discovery: "
                f"{sorted(discovered_tools)}"
            )

        customer = await runtime.invoke(
            "find_customer",
            {"email": "lara.neumann@example.test"},
        )
        if not customer.get("ok") or not customer.get("found"):
            raise RuntimeError("Lara Neumann was not found through CRM MCP.")

        customer_id = customer["customer"]["id"]
        policies = await runtime.invoke(
            "get_customer_policies",
            {"customer_id": customer_id},
        )
        policy = await runtime.invoke(
            "get_policy",
            {"policy_number": "TEST-KFZ-2026-1001"},
        )
        claims = await runtime.invoke(
            "get_customer_claims",
            {"customer_id": customer_id},
        )
        claim = await runtime.invoke(
            "get_claim_status",
            {"claim_number": "TEST-CLM-2026-2001"},
        )

        if not policy.get("ok") or not policy.get("found"):
            raise RuntimeError("The Lara Neumann policy was not found.")
        if not claim.get("ok") or not claim.get("found"):
            raise RuntimeError("The Lara Neumann claim was not found.")

        print(
            json.dumps(
                {
                    "tools": sorted(discovered_tools),
                    "customer": customer,
                    "customer_policies": policies,
                    "policy": policy,
                    "customer_claims": claims,
                    "claim": claim,
                },
                indent=2,
            )
        )
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
