import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src import main
from src.api.rag_service import AnswerResult, Source as RagSource


class FakeRuntime:
    def __init__(self):
        self.invoke = AsyncMock(side_effect=self._invoke)

    async def _invoke(self, tool_name, arguments):
        if tool_name == "find_customer":
            return {
                "ok": True,
                "available": True,
                "found": True,
                "customer": {
                    "id": "contact-1",
                    "display_name": "Lara Neumann",
                    "email_hint": "l***@example.test",
                },
            }
        if tool_name == "get_customer_policies":
            return {
                "ok": True,
                "available": True,
                "count": 1,
                "policies": [
                    {
                        "id": "policy-1",
                        "policy_number": "TEST-KFZ-2026-1001",
                        "product_type": "Helvetia Motor Vehicle Insurance",
                        "coverage_type": "Partially comprehensive cover",
                        "status": "Active",
                        "deductible": {"amount": 150, "currency": "EUR"},
                        "annual_premium": {"amount": 684, "currency": "EUR"},
                        "start_date": "2026-01-01",
                        "end_date": "2026-12-31",
                    }
                ],
            }
        if tool_name == "get_customer_claims":
            return {
                "ok": True,
                "available": True,
                "count": 1,
                "claims": [
                    {
                        "id": "claim-1",
                        "claim_number": "TEST-CLM-2026-2001",
                        "claim_date": "2026-07-10",
                        "damage_type": "Glass Damage",
                        "claimed_amount": {"amount": 780, "currency": "EUR"},
                        "status": "Under Review",
                        "policy_reference": "TEST-KFZ-2026-1001",
                    }
                ],
            }
        raise AssertionError(f"Unexpected tool call: {tool_name} {arguments}")


class CRMApiRoutingTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_crm_question_returns_structured_disabled_error(self):
        with patch.object(main, "_crm_runtime_ready", False), patch.object(
            main,
            "_crm_runtime_error",
            {
                "category": "disabled",
                "message": "CRM integration is disabled.",
            },
        ):
            response = self.client.post(
                "/api/ask",
                json={"question": "Which active policies does Lara Neumann have?"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["category"], "disabled")

    def test_crm_only_route_does_not_call_rag(self):
        runtime = FakeRuntime()
        with patch.object(main, "_crm_runtime_ready", True), patch.object(
            main, "get_mcp_runtime", return_value=runtime
        ), patch.object(main.rag_service, "run_rag") as rag:
            response = self.client.post(
                "/api/ask",
                json={"question": "Which active policies does Lara Neumann have?"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("CRM facts", body["answer"])
        self.assertIn("TEST-KFZ-2026-1001", body["answer"])
        self.assertTrue(
            any(source["documentId"] == "espocrm:policy-1" for source in body["sources"])
        )
        rag.assert_not_called()

    def test_crm_only_current_policy_uses_metadata_selection(self):
        runtime = FakeRuntime()
        original_invoke = runtime._invoke

        async def invoke_with_policy_history(tool_name, arguments):
            if tool_name != "get_customer_policies":
                return await original_invoke(tool_name, arguments)
            return {
                "ok": True,
                "available": True,
                "count": 2,
                "policies": [
                    {
                        "id": "policy-old",
                        "policy_number": "TEST-KFZ-2026-1001",
                        "product_type": "Helvetia Motor Vehicle Insurance",
                        "coverage_type": "Partially comprehensive cover",
                        "status": "Active",
                        "deductible": {"amount": 150, "currency": "EUR"},
                        "annual_premium": {"amount": 684, "currency": "EUR"},
                        "start_date": "2026-01-01",
                        "end_date": "2026-12-31",
                    },
                    {
                        "id": "policy-current",
                        "policy_number": "TEST-KFZ-2026-1003",
                        "product_type": "Helvetia Motor Vehicle Insurance",
                        "coverage_type": "Partially comprehensive cover",
                        "status": "Active",
                        "deductible": {"amount": 300, "currency": "EUR"},
                        "annual_premium": {"amount": 720, "currency": "EUR"},
                        "start_date": "2026-07-01",
                        "end_date": "2027-06-30",
                    },
                ],
            }

        runtime.invoke = AsyncMock(side_effect=invoke_with_policy_history)
        with patch.object(main, "_crm_runtime_ready", True), patch.object(
            main, "get_mcp_runtime", return_value=runtime
        ), patch.object(main.rag_service, "run_rag") as rag:
            response = self.client.post(
                "/api/ask",
                json={
                    "question": (
                        "What is Lara Neumann's current motor insurance policy "
                        "as of 2026-08-01?"
                    )
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("TEST-KFZ-2026-1003", body["answer"])
        self.assertNotIn("TEST-KFZ-2026-1001", body["answer"])
        self.assertEqual(
            [source["section"] for source in body["sources"] if "Policy" in source["documentTitle"]],
            ["TEST-KFZ-2026-1003"],
        )
        rag.assert_not_called()

    def test_retrieval_only_route_does_not_call_crm_when_crm_is_unavailable(self):
        runtime = FakeRuntime()
        document_result = AnswerResult(
            answer="Partial coverage is defined in the insurance conditions.",
            sources=[],
            query="q",
            latency_ms=10,
        )
        with patch.object(main, "_crm_runtime_ready", False), patch.object(
            main,
            "_crm_runtime_error",
            {"category": "connection", "message": "CRM unavailable."},
        ), patch.object(
            main, "get_mcp_runtime", return_value=runtime
        ), patch.object(
            main.rag_service, "run_rag", return_value=document_result
        ) as rag:
            response = self.client.post(
                "/api/ask",
                json={"question": "What does partial coverage generally cover?"},
            )

        self.assertEqual(response.status_code, 200)
        rag.assert_called_once()
        runtime.invoke.assert_not_awaited()

    def test_combined_route_sends_selected_chunks_and_crm_data_to_one_generation(self):
        runtime = FakeRuntime()
        document_result = AnswerResult(
            answer=(
                "CRM facts: claim TEST-CLM-2026-2001 is Under Review. "
                "Document evidence: the wording covers specified glass damage "
                "[conditions.pdf:12]."
            ),
            sources=[
                RagSource(
                    document_id="espocrm:contact-1",
                    document_title="EspoCRM Contact",
                    page=None,
                    section="Lara Neumann",
                    snippet="Customer: Lara Neumann.",
                ),
                RagSource(
                    document_id="espocrm:policy-1",
                    document_title="EspoCRM Policy",
                    page=None,
                    section="TEST-KFZ-2026-1001",
                    snippet=(
                        "Policy TEST-KFZ-2026-1001: Partially comprehensive cover."
                    ),
                ),
                RagSource(
                    document_id="espocrm:claim-1",
                    document_title="EspoCRM Claim",
                    page=None,
                    section="TEST-CLM-2026-2001",
                    snippet="Claim TEST-CLM-2026-2001: status Under Review.",
                ),
                RagSource(
                    document_id="conditions.pdf",
                    document_title="Motor Conditions",
                    page=12,
                    section="Glass",
                    snippet="Specified glass damage is covered.",
                )
            ],
            query="q",
            latency_ms=30,
        )
        with patch.object(main, "_crm_runtime_ready", True), patch.object(
            main, "get_mcp_runtime", return_value=runtime
        ), patch.object(
            main.rag_service,
            "run_rag",
            return_value=document_result,
        ) as rag:
            response = self.client.post(
                "/api/ask",
                json={
                    "question": (
                        "Lara Neumann reported glass damage. Is the damage "
                        "covered by her current policy, and what is the current "
                        "claim status?"
                    )
                },
            )

        self.assertEqual(response.status_code, 200)
        answer = response.json()["answer"]
        self.assertIn("CRM facts:", answer)
        self.assertIn("Document evidence:", answer)
        self.assertIn("Under Review", answer)
        self.assertEqual(len(response.json()["sources"]), 4)
        self.assertNotIn("Inference:", answer)

        rag.assert_called_once()
        call_kwargs = rag.call_args.kwargs
        self.assertIn("glass damage", call_kwargs["retrieval_query"])
        self.assertNotIn("Lara Neumann", call_kwargs["retrieval_query"])
        crm_context = call_kwargs["additional_context_docs"]
        self.assertEqual(len(crm_context), 3)
        self.assertTrue(
            all(
                document.metadata["source_type"] == "crm"
                for document in crm_context
            )
        )
        self.assertTrue(
            all(
                document.page_content.startswith("CRM FACT\n")
                for document in crm_context
            )
        )

    def test_windscreen_query_sends_only_latest_relevant_motor_policy(self):
        runtime = FakeRuntime()
        original_invoke = runtime._invoke

        async def invoke_with_multiple_policies(tool_name, arguments):
            if tool_name != "get_customer_policies":
                return await original_invoke(tool_name, arguments)
            return {
                "ok": True,
                "available": True,
                "count": 3,
                "policies": [
                    {
                        "id": "policy-new",
                        "name": (
                            "Lara Neumann Helvetia Motor Vehicle Insurance "
                            "Partially comprehensive cover - STI Edition March 2026"
                        ),
                        "policy_number": "TEST-KFZ-2026-1003",
                        "product_type": "Helvetia Motor Vehicle Insurance",
                        "coverage_type": "Partially comprehensive cover",
                        "status": "Active",
                        "deductible": {"amount": 300, "currency": "EUR"},
                        "annual_premium": {"amount": 720, "currency": "EUR"},
                        "start_date": "2026-07-01",
                        "end_date": "2027-06-30",
                    },
                    {
                        "id": "policy-liability",
                        "name": "Lara Neumann Liability 2026",
                        "policy_number": "TEST-PHV-2026-1002",
                        "product_type": "Personal Liability",
                        "coverage_type": "Liability",
                        "status": "Active",
                        "start_date": "2026-01-01",
                        "end_date": "2026-12-31",
                    },
                    {
                        "id": "policy-old",
                        "name": (
                            "Lara Neumann Helvetia Motor Vehicle Insurance "
                            "Partially comprehensive cover - STI Edition March 2026"
                        ),
                        "policy_number": "TEST-KFZ-2026-1001",
                        "product_type": "Helvetia Motor Vehicle Insurance",
                        "coverage_type": "Partially comprehensive cover",
                        "status": "Active",
                        "deductible": {"amount": 150, "currency": "EUR"},
                        "annual_premium": {"amount": 684, "currency": "EUR"},
                        "start_date": "2026-01-01",
                        "end_date": "2026-12-31",
                    },
                ],
            }

        runtime.invoke = AsyncMock(side_effect=invoke_with_multiple_policies)
        document_result = AnswerResult(
            answer=(
                "The windscreen rule is conditional "
                "[motor-vehicle-insurance-sti.pdf, page 9]."
            ),
            sources=[
                RagSource(
                    document_id="espocrm:policy-new",
                    document_title="EspoCRM Policy",
                    section="TEST-KFZ-2026-1003",
                    snippet=(
                        "Policy TEST-KFZ-2026-1003: Partially comprehensive cover."
                    ),
                ),
                RagSource(
                    document_id="motor-vehicle-insurance-sti",
                    document_title="motor-vehicle-insurance-sti.pdf",
                    page=7,
                    snippet="The deductible will not be applied for a repair.",
                ),
            ],
            query="q",
            latency_ms=30,
        )
        question = (
            "Lara Neumann reports damage to the windscreen of her car. Is "
            "this damage generally covered by her current motor insurance, "
            "and would she still have to pay her individual deductible if "
            "the windscreen can be repaired instead of replaced? Please "
            "provide the relevant active policy number, coverage type, "
            "individual deductible and annual premium. Clearly distinguish "
            "the general insurance terms from Lara's individual contract "
            "data, and do not make a final claim decision."
        )
        with patch.object(main, "_crm_runtime_ready", True), patch.object(
            main, "get_mcp_runtime", return_value=runtime
        ), patch.object(
            main.rag_service,
            "run_rag",
            return_value=document_result,
        ) as rag:
            response = self.client.post(
                "/api/ask",
                json={"question": question},
            )

        self.assertEqual(response.status_code, 200)
        called_tools = [
            call.args[0] for call in runtime.invoke.await_args_list
        ]
        self.assertNotIn("get_customer_claims", called_tools)
        call_kwargs = rag.call_args.kwargs
        crm_context = call_kwargs["additional_context_docs"]
        self.assertEqual(len(crm_context), 2)
        context_text = "\n".join(doc.page_content for doc in crm_context)
        self.assertIn("TEST-KFZ-2026-1003", context_text)
        self.assertNotIn("TEST-KFZ-2026-1001", context_text)
        self.assertNotIn("TEST-PHV-2026-1002", context_text)
        self.assertIn("Helvetia Motor Vehicle Insurance", call_kwargs["retrieval_query"])
        self.assertIn("STI Edition March 2026", call_kwargs["retrieval_query"])
        self.assertNotIn("annual premium", call_kwargs["retrieval_query"])
        self.assertEqual(
            response.json()["crmResult"]["selection"]["selectedPolicyNumbers"],
            ["TEST-KFZ-2026-1003"],
        )

    def test_broad_crm_export_is_forbidden(self):
        response = self.client.post(
            "/api/ask",
            json={"question": "Export all customers from CRM."},
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            response.json()["detail"]["category"],
            "unauthorized_scope",
        )


if __name__ == "__main__":
    unittest.main()
