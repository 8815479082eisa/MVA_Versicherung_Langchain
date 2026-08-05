import json
import unittest

import httpx

from src.integrations.espocrm_client import (
    CRMClientError,
    EspoCRMClient,
    EspoCRMConfig,
)


def _config(**overrides):
    values = {
        "enabled": True,
        "base_url": "https://crm.test",
        "api_key": "unit-test-key",
        "request_timeout_seconds": 0.2,
        "max_pages": 3,
        "page_size": 2,
    }
    values.update(overrides)
    return EspoCRMConfig(**values)


class EspoCRMClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_find_customer_uses_get_and_normalizes_minimal_fields(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "total": 1,
                    "list": [
                        {
                            "id": "contact-1",
                            "firstName": "Lara",
                            "lastName": "Neumann",
                            "emailAddress": "lara.neumann@example.test",
                            "phoneNumber": "must-not-leak",
                        }
                    ],
                },
            )

        async with EspoCRMClient(
            _config(), transport=httpx.MockTransport(handler)
        ) as client:
            result = await client.find_customer(name="Lara Neumann")

        self.assertTrue(result["found"])
        self.assertEqual(result["customer"]["display_name"], "Lara Neumann")
        self.assertEqual(result["customer"]["email"], "lara.neumann@example.test")
        self.assertEqual(result["customer"]["phone"], "must-not-leak")
        self.assertTrue(all(request.method == "GET" for request in requests))
        search = json.loads(requests[0].url.params["searchParams"])
        self.assertIn("emailAddress", search["select"])
        self.assertIn("phoneNumber", search["select"])
        self.assertIn("birthDate", search["select"])

    async def test_no_customer_is_explicit(self):
        transport = httpx.MockTransport(
            lambda _: httpx.Response(200, json={"total": 0, "list": []})
        )
        async with EspoCRMClient(_config(), transport=transport) as client:
            result = await client.find_customer(email="nobody@example.test")
        self.assertEqual(result, {"found": False, "customer": None})

    async def test_duplicate_customer_requires_clarification(self):
        transport = httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "total": 2,
                    "list": [
                        {"id": "a", "firstName": "Lara", "lastName": "Neumann"},
                        {"id": "b", "firstName": "Lara", "lastName": "Neumann"},
                    ],
                },
            )
        )
        async with EspoCRMClient(_config(), transport=transport) as client:
            result = await client.find_customer(name="Lara Neumann")
        self.assertFalse(result["found"])
        self.assertTrue(result["ambiguous"])
        self.assertEqual(result["match_count"], 2)

    async def test_policy_pagination_and_normalization(self):
        offsets = []

        def handler(request: httpx.Request) -> httpx.Response:
            search = json.loads(request.url.params["searchParams"])
            offsets.append(search["offset"])
            rows = [
                {
                    "id": f"p-{index}",
                    "policyNumber": f"TEST-KFZ-2026-10{index:02d}",
                    "productType": "Helvetia Motor Vehicle Insurance",
                    "coverageType": "Partially comprehensive cover",
                    "status": "Active",
                    "deductible": "150.00",
                    "annualPremium": "684.50",
                    "currency": "EUR",
                    "startDate": "2026-01-01",
                    "endDate": "2026-12-31",
                }
                for index in range(1, 4)
            ]
            offset = search["offset"]
            return httpx.Response(
                200,
                json={"total": 3, "list": rows[offset : offset + search["maxSize"]]},
            )

        async with EspoCRMClient(
            _config(), transport=httpx.MockTransport(handler)
        ) as client:
            result = await client.get_customer_policies("contact-1")

        self.assertEqual(offsets, [0, 2])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["policies"][0]["deductible"]["amount"], 150)
        self.assertEqual(result["policies"][0]["annual_premium"]["amount"], 684.5)

    async def test_exact_claim_lookup(self):
        transport = httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "total": 1,
                    "list": [
                        {
                            "id": "claim-1",
                            "claimNumber": "TEST-CLM-2026-2001",
                            "claimDate": "2026-07-10",
                            "damageType": "Glass Damage",
                            "claimedAmount": 780,
                            "currency": "EUR",
                            "status": "Under Review",
                            "policyName": "TEST-KFZ-2026-1001",
                        }
                    ],
                },
            )
        )
        async with EspoCRMClient(_config(), transport=transport) as client:
            result = await client.get_claim_status("test-clm-2026-2001")
        self.assertTrue(result["found"])
        self.assertEqual(result["claim"]["status"], "Under Review")

    async def test_exact_policy_lookup(self):
        transport = httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "total": 1,
                    "list": [
                        {
                            "id": "policy-1",
                            "policyNumber": "TEST-KFZ-2026-1001",
                            "productType": "Helvetia Motor Vehicle Insurance",
                            "coverageType": "Partially comprehensive cover",
                            "status": "Active",
                            "deductible": 150,
                            "annualPremium": 684,
                            "currency": "EUR",
                            "startDate": "2026-01-01",
                            "endDate": "2026-12-31",
                        }
                    ],
                },
            )
        )
        async with EspoCRMClient(_config(), transport=transport) as client:
            result = await client.get_policy("TEST-KFZ-2026-1001")
        self.assertTrue(result["found"])
        self.assertEqual(
            result["policy"]["coverage_type"],
            "Partially comprehensive cover",
        )

    async def test_authentication_error_is_sanitized(self):
        transport = httpx.MockTransport(lambda _: httpx.Response(403, text="secret"))
        async with EspoCRMClient(_config(), transport=transport) as client:
            with self.assertRaises(CRMClientError) as caught:
                await client.get_policy("TEST-KFZ-2026-1001")
        self.assertEqual(caught.exception.category, "authentication")
        self.assertNotIn("secret", str(caught.exception))

    async def test_timeout_retries_once(self):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            raise httpx.ReadTimeout("late", request=request)

        async with EspoCRMClient(
            _config(), transport=httpx.MockTransport(handler)
        ) as client:
            with self.assertRaises(CRMClientError) as caught:
                await client.get_policy("TEST-KFZ-2026-1001")
        self.assertEqual(caught.exception.category, "timeout")
        self.assertEqual(attempts, 2)

    async def test_connection_error_retries_once(self):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            raise httpx.ConnectError("offline", request=request)

        async with EspoCRMClient(
            _config(), transport=httpx.MockTransport(handler)
        ) as client:
            with self.assertRaises(CRMClientError) as caught:
                await client.get_policy("TEST-KFZ-2026-1001")
        self.assertEqual(caught.exception.category, "connection")
        self.assertEqual(attempts, 2)

    async def test_malformed_json_is_rejected(self):
        transport = httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                content=b"not-json",
                headers={"content-type": "application/json"},
            )
        )
        async with EspoCRMClient(_config(), transport=transport) as client:
            with self.assertRaises(CRMClientError) as caught:
                await client.get_policy("TEST-KFZ-2026-1001")
        self.assertEqual(caught.exception.category, "malformed_response")

    async def test_disabled_client_performs_no_request(self):
        calls = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json={})

        async with EspoCRMClient(
            _config(enabled=False),
            transport=httpx.MockTransport(handler),
        ) as client:
            with self.assertRaises(CRMClientError) as caught:
                await client.get_policy("TEST-KFZ-2026-1001")
        self.assertEqual(caught.exception.category, "disabled")
        self.assertEqual(calls, 0)


if __name__ == "__main__":
    unittest.main()
