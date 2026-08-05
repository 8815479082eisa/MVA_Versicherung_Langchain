import unittest
from unittest.mock import AsyncMock, patch

from src.integrations.espocrm_client import CRMClientError
from src.mcp_servers import crm_server


class CRMToolTest(unittest.IsolatedAsyncioTestCase):
    async def test_server_exposes_exactly_five_read_only_tools(self):
        tools = await crm_server.mcp.list_tools()
        names = {tool.name for tool in tools}
        self.assertEqual(
            names,
            {
                "find_customer",
                "get_customer_policies",
                "get_policy",
                "get_customer_claims",
                "get_claim_status",
            },
        )
        self.assertFalse(
            any(
                token in name
                for name in names
                for token in ("create", "update", "delete", "write", "export", "list_all")
            )
        )
        schemas = {tool.name: tool.parameters for tool in tools}
        self.assertEqual(
            set(schemas["find_customer"]["properties"]),
            {"name", "email"},
        )
        self.assertEqual(
            schemas["get_customer_policies"]["required"],
            ["customer_id"],
        )
        self.assertEqual(
            schemas["get_policy"]["required"],
            ["policy_number"],
        )
        self.assertEqual(
            schemas["get_customer_claims"]["required"],
            ["customer_id"],
        )
        self.assertEqual(
            schemas["get_claim_status"]["required"],
            ["claim_number"],
        )
        self.assertTrue(
            all(
                schema.get("additionalProperties") is False
                for schema in schemas.values()
            )
        )

    async def test_success_response_is_bounded(self):
        fake_client = AsyncMock()
        fake_client.get_policy.return_value = {
            "found": True,
            "policy": {
                "id": "p-1",
                "policy_number": "TEST-KFZ-2026-1001",
                "status": "Active",
            },
        }
        with patch.object(crm_server, "_crm_client", fake_client):
            result = await crm_server._get_policy_impl("TEST-KFZ-2026-1001")

        self.assertTrue(result["ok"])
        self.assertTrue(result["available"])
        fake_client.get_policy.assert_awaited_once_with("TEST-KFZ-2026-1001")

    async def test_availability_error_is_structured(self):
        fake_client = AsyncMock()
        fake_client.get_policy.side_effect = CRMClientError(
            "connection",
            "EspoCRM is unavailable or the connection failed.",
        )
        with patch.object(crm_server, "_crm_client", fake_client):
            result = await crm_server._get_policy_impl("TEST-KFZ-2026-1001")

        self.assertFalse(result["ok"])
        self.assertFalse(result["available"])
        self.assertEqual(result["error"]["category"], "connection")

    async def test_unexpected_error_does_not_leak_details(self):
        fake_client = AsyncMock()
        fake_client.get_policy.side_effect = RuntimeError("private-token-value")
        with patch.object(crm_server, "_crm_client", fake_client):
            result = await crm_server._get_policy_impl("TEST-KFZ-2026-1001")

        self.assertEqual(result["error"]["category"], "internal")
        self.assertNotIn("private-token-value", str(result))


if __name__ == "__main__":
    unittest.main()
