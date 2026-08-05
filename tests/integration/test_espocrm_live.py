import os
import unittest

from src.integrations.espocrm_client import EspoCRMClient, EspoCRMConfig
from src.mcp_clients.insurance_tools import PersistentMCPToolRuntime


def _live_config():
    base_url = os.getenv("ESPOCRM_PUBLIC_URL", "").strip()
    api_key = os.getenv("ESPOCRM_API_KEY", "").strip()
    enabled = os.getenv("CRM_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not (enabled and base_url and api_key):
        return None
    return EspoCRMConfig(
        enabled=True,
        base_url=base_url,
        api_key=api_key,
        request_timeout_seconds=5,
        max_pages=10,
        page_size=50,
    )


@unittest.skipUnless(
    _live_config() is not None,
    "Live EspoCRM smoke test requires CRM_ENABLED=true, ESPOCRM_PUBLIC_URL, and ESPOCRM_API_KEY.",
)
class EspoCRMLiveTest(unittest.IsolatedAsyncioTestCase):
    async def test_rest_and_mcp_return_seeded_lara_records(self):
        config = _live_config()
        assert config is not None
        async with EspoCRMClient(config) as client:
            health = await client.health()
            self.assertTrue(health["authenticated"])
            customer = await client.find_customer(
                email="lara.neumann@example.test"
            )
            self.assertTrue(customer["found"])
            policy = await client.get_policy("TEST-KFZ-2026-1001")
            self.assertTrue(policy["found"])
            self.assertEqual(policy["policy"]["status"], "Active")
            claim = await client.get_claim_status("TEST-CLM-2026-2001")
            self.assertTrue(claim["found"])
            self.assertEqual(claim["claim"]["status"], "Under Review")

        original_url = os.environ.get("ESPOCRM_BASE_URL")
        os.environ["ESPOCRM_BASE_URL"] = config.base_url
        runtime = PersistentMCPToolRuntime()
        try:
            await runtime.start(("insurance_crm",))
            result = await runtime.invoke(
                "get_policy",
                {"policy_number": "TEST-KFZ-2026-1001"},
            )
            self.assertTrue(result["ok"])
            self.assertTrue(result["found"])
        finally:
            await runtime.close()
            if original_url is None:
                os.environ.pop("ESPOCRM_BASE_URL", None)
            else:
                os.environ["ESPOCRM_BASE_URL"] = original_url


if __name__ == "__main__":
    unittest.main()
