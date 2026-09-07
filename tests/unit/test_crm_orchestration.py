import unittest
from datetime import date

from src.core.crm_orchestration import (
    CRMOrchestrationError,
    CRMQueryResult,
    execute_crm_query,
    select_crm_context,
    select_crm_sources_for_context,
)
from src.core.insurance_tool_routing import plan_insurance_query


class FakeInvoker:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def __call__(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return self.responses[tool_name]


class CRMOrchestrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_customer_lookup_precedes_bounded_policy_read(self):
        invoker = FakeInvoker(
            {
                "find_customer": {
                    "ok": True,
                    "available": True,
                    "found": True,
                    "customer": {
                        "id": "contact-1",
                        "display_name": "Lara Neumann",
                        "email_hint": "l***@example.test",
                    },
                },
                "get_customer_policies": {
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
                },
            }
        )
        result = await execute_crm_query(
            plan_insurance_query("Which active policies does Lara Neumann have?"),
            invoker,
        )

        self.assertEqual(invoker.calls[0][0], "find_customer")
        self.assertEqual(invoker.calls[1][0], "get_customer_policies")
        self.assertIn("TEST-KFZ-2026-1001", result.answer)
        self.assertEqual(len(result.sources), 2)

    async def test_no_customer_does_not_issue_dependent_reads(self):
        invoker = FakeInvoker(
            {
                "find_customer": {
                    "ok": True,
                    "available": True,
                    "found": False,
                    "customer": None,
                }
            }
        )
        result = await execute_crm_query(
            plan_insurance_query("Which policies does Nobody Example have?"),
            invoker,
        )
        self.assertEqual([name for name, _ in invoker.calls], ["find_customer"])
        self.assertIn("No matching customer", result.answer)

    async def test_contact_only_query_returns_authorized_contact_fields(self):
        invoker = FakeInvoker({
            "find_customer": {
                "ok": True,
                "available": True,
                "found": True,
                "customer": {
                    "id": "contact-1",
                    "display_name": "Lara Neumann",
                    "address": "Teststrasse 17, 12345 Berlin",
                    "phone": "+49 151 23456789",
                    "email": "lara.neumann@example.test",
                    "date_of_birth": "1988-05-14",
                },
            }
        })
        result = await execute_crm_query(
            plan_insurance_query(
                "What is Lara Neumann's address, telephone number and email address?"
            ),
            invoker,
        )
        self.assertEqual([name for name, _ in invoker.calls], ["find_customer"])
        self.assertIn("Teststrasse 17", result.answer)
        self.assertIn("+49 151 23456789", result.answer)
        self.assertIn("lara.neumann@example.test", result.answer)

    async def test_ambiguous_customer_returns_clarification(self):
        invoker = FakeInvoker({
            "find_customer": {
                "ok": True,
                "available": True,
                "found": False,
                "ambiguous": True,
                "match_count": 2,
                "customer": None,
            }
        })
        result = await execute_crm_query(
            plan_insurance_query("Show Lara Neumann's policies."),
            invoker,
        )
        self.assertEqual(result.reason_code, "AMBIGUOUS_CUSTOMER")
        self.assertIn("provide the customer number or policy number", result.answer)

    async def test_entity_mismatch_is_rejected(self):
        invoker = FakeInvoker({
            "find_customer": {
                "ok": True,
                "available": True,
                "found": True,
                "customer": {"id": "contact-lara", "display_name": "Lara Neumann"},
            },
            "get_customer_policies": {
                "ok": True,
                "available": True,
                "count": 1,
                "policies": [{
                    "id": "policy-other",
                    "policy_number": "TEST-KFZ-2026-9999",
                    "customer_id": "contact-other",
                }],
            },
        })
        with self.assertRaises(CRMOrchestrationError) as caught:
            await execute_crm_query(
                plan_insurance_query("Show Lara Neumann's policies."),
                invoker,
            )
        self.assertEqual(caught.exception.error["reason_code"], "ENTITY_MISMATCH")

    async def test_tool_availability_failure_is_propagated(self):
        invoker = FakeInvoker(
            {
                "get_claim_status": {
                    "ok": False,
                    "available": False,
                    "error": {
                        "category": "timeout",
                        "message": "CRM timed out.",
                    },
                }
            }
        )
        with self.assertRaises(CRMOrchestrationError) as caught:
            await execute_crm_query(
                plan_insurance_query(
                    "Show claim status for TEST-CLM-2026-2001."
                ),
                invoker,
            )
        self.assertFalse(caught.exception.available)
        self.assertEqual(caught.exception.error["category"], "timeout")

    def test_combined_context_selects_contact_and_relevant_policy(self):
        result = CRMQueryResult(
            answer="all facts",
            sources=(
                {
                    "documentId": "espocrm:contact-1",
                    "documentTitle": "EspoCRM Contact",
                    "section": "Lara Neumann",
                    "snippet": "Customer: Lara Neumann.",
                },
                {
                    "documentId": "espocrm:old-motor",
                    "documentTitle": "EspoCRM Policy",
                    "section": "TEST-KFZ-2026-1001",
                    "snippet": (
                        "Policy TEST-KFZ-2026-1001: Helvetia Motor Vehicle "
                        "Insurance, Partially comprehensive cover, Active; "
                        "deductible 150 EUR."
                    ),
                },
                {
                    "documentId": "espocrm:helvetia-current",
                    "documentTitle": "EspoCRM Policy",
                    "section": "TEST-KFZ-2026-1003",
                    "snippet": (
                        "Policy TEST-KFZ-2026-1003 (Lara Neumann "
                        "Helvetia Motor Vehicle Insurance Partially "
                        "comprehensive cover - STI Edition March 2026): "
                        "Helvetia Motor Vehicle Insurance, Partially "
                        "comprehensive cover, Active; "
                        "deductible 300 EUR."
                    ),
                },
            ),
            tool_results=(),
        )

        selected = select_crm_sources_for_context(
            result,
            (
                "For Lara Neumann's Helvetia Motor Vehicle policy, "
                "are marten bites covered?"
            ),
        )

        self.assertEqual(
            {source["documentId"] for source in selected},
            {"espocrm:contact-1", "espocrm:helvetia-current"},
        )

    def test_current_motor_query_selects_latest_effective_active_policy_only(self):
        policies = [
            {
                "id": "new-motor",
                "name": (
                    "Lara Neumann Helvetia Motor Vehicle Insurance Partially "
                    "comprehensive cover - STI Edition March 2026"
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
                "id": "liability",
                "name": "Lara Neumann Liability 2026",
                "policy_number": "TEST-PHV-2026-1002",
                "product_type": "Personal Liability",
                "coverage_type": "Liability",
                "status": "Active",
                "start_date": "2026-01-01",
                "end_date": "2026-12-31",
            },
            {
                "id": "old-motor",
                "name": (
                    "Lara Neumann Helvetia Motor Vehicle Insurance Partially "
                    "comprehensive cover - STI Edition March 2026"
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
        ]
        sources = (
            {
                "documentId": "espocrm:contact-1",
                "documentTitle": "EspoCRM Contact",
                "section": "Lara Neumann",
                "snippet": "Customer: Lara Neumann.",
            },
            *(
                {
                    "documentId": f"espocrm:{policy['id']}",
                    "documentTitle": "EspoCRM Policy",
                    "section": policy["policy_number"],
                    "snippet": (
                        f"Policy {policy['policy_number']} ({policy['name']}): "
                        f"{policy['product_type']}, {policy['coverage_type']}, "
                        f"{policy['status']}; term {policy['start_date']} to "
                        f"{policy['end_date']}."
                    ),
                }
                for policy in policies
            ),
        )
        result = CRMQueryResult(
            answer="all facts",
            sources=sources,
            tool_results=(
                {
                    "found": True,
                    "customer": {
                        "id": "contact-1",
                        "display_name": "Lara Neumann",
                    },
                },
                {"count": 3, "policies": policies},
            ),
        )

        selection = select_crm_context(
            result,
            (
                "Lara Neumann reports windscreen damage to her car. Is it "
                "covered by her current motor insurance and what is her "
                "individual deductible?"
            ),
            as_of_date=date(2026, 7, 29),
        )

        self.assertEqual(
            selection.selected_policy_numbers,
            ("TEST-KFZ-2026-1003",),
        )
        self.assertEqual(
            {
                source["documentId"]
                for source in selection.sources
                if source["documentTitle"] == "EspoCRM Policy"
            },
            {"espocrm:new-motor"},
        )
        selected_evidence = [
            item for item in selection.evidence if item.get("selected")
        ]
        self.assertEqual(selected_evidence[0]["policy_number"], "TEST-KFZ-2026-1003")
        self.assertTrue(selected_evidence[0]["signals"]["active_status"])
        self.assertTrue(selected_evidence[0]["signals"]["temporally_valid"])
        hints_text = " ".join(selection.retrieval_hints)
        self.assertIn("Helvetia Motor Vehicle Insurance", hints_text)
        self.assertIn("STI Edition March 2026", hints_text)

    def test_explicit_policy_number_overrides_recency(self):
        result = CRMQueryResult(
            answer="all facts",
            sources=(
                {
                    "documentId": "espocrm:old",
                    "documentTitle": "EspoCRM Policy",
                    "section": "TEST-KFZ-2026-1001",
                    "snippet": (
                        "Policy TEST-KFZ-2026-1001: Helvetia Motor Vehicle "
                        "Insurance, Active."
                    ),
                },
                {
                    "documentId": "espocrm:new",
                    "documentTitle": "EspoCRM Policy",
                    "section": "TEST-KFZ-2026-1003",
                    "snippet": (
                        "Policy TEST-KFZ-2026-1003: Helvetia Motor Vehicle "
                        "Insurance, Active."
                    ),
                },
            ),
            tool_results=(
                {
                    "policies": [
                        {
                            "id": "old",
                            "policy_number": "TEST-KFZ-2026-1001",
                            "product_type": "Helvetia Motor Vehicle Insurance",
                            "status": "Active",
                            "start_date": "2026-01-01",
                            "end_date": "2026-12-31",
                        },
                        {
                            "id": "new",
                            "policy_number": "TEST-KFZ-2026-1003",
                            "product_type": "Helvetia Motor Vehicle Insurance",
                            "status": "Active",
                            "start_date": "2026-07-01",
                            "end_date": "2027-06-30",
                        },
                    ]
                },
            ),
        )

        selection = select_crm_context(
            result,
            "Show policy TEST-KFZ-2026-1001 for this motor contract.",
            as_of_date=date(2026, 7, 29),
        )

        self.assertEqual(
            selection.selected_policy_numbers,
            ("TEST-KFZ-2026-1001",),
        )

    def test_comparison_query_preserves_old_and_current_policy_context(self):
        policies = [
            {
                "id": "old",
                "policy_number": "TEST-KFZ-2026-1001",
                "product_type": "Helvetia Motor Vehicle Insurance",
                "status": "Active",
                "start_date": "2026-01-01",
                "end_date": "2026-12-31",
            },
            {
                "id": "current",
                "policy_number": "TEST-KFZ-2026-1003",
                "product_type": "Helvetia Motor Vehicle Insurance",
                "status": "Active",
                "start_date": "2026-07-01",
                "end_date": "2027-06-30",
            },
        ]
        result = CRMQueryResult(
            answer="all facts",
            sources=tuple(
                {
                    "documentId": f"espocrm:{policy['id']}",
                    "documentTitle": "EspoCRM Policy",
                    "section": policy["policy_number"],
                    "snippet": (
                        f"Policy {policy['policy_number']}: Helvetia Motor "
                        "Vehicle Insurance, Active."
                    ),
                }
                for policy in policies
            ),
            tool_results=({"policies": policies},),
        )

        selection = select_crm_context(
            result,
            "How did Lara's premium change between her previous and current policy?",
            as_of_date=date(2026, 8, 1),
        )

        self.assertEqual(
            set(selection.selected_policy_numbers),
            {"TEST-KFZ-2026-1001", "TEST-KFZ-2026-1003"},
        )

    def test_plural_claim_question_preserves_all_customer_claims(self):
        claims = (
            {
                "documentId": "espocrm:claim-1",
                "documentTitle": "EspoCRM Claim",
                "section": "TEST-CLM-2026-2601",
                "snippet": "Claim TEST-CLM-2026-2601: Collision, status Approved.",
            },
            {
                "documentId": "espocrm:claim-2",
                "documentTitle": "EspoCRM Claim",
                "section": "TEST-CLM-2026-2602",
                "snippet": "Claim TEST-CLM-2026-2602: Property Damage, status Closed.",
            },
        )
        result = CRMQueryResult(
            answer="all facts",
            sources=claims,
            tool_results=(
                {
                    "claims": [
                        {"id": "claim-1", "claim_date": "2026-06-18"},
                        {"id": "claim-2", "claim_date": "2026-07-08"},
                    ]
                },
            ),
        )

        selection = select_crm_context(
            result,
            "Which claims are recorded for Leon Becker?",
        )

        self.assertEqual(
            {source["documentId"] for source in selection.sources},
            {"espocrm:claim-1", "espocrm:claim-2"},
        )

    def test_contact_source_is_minimized_when_contact_fields_are_not_requested(self):
        result = CRMQueryResult(
            answer="facts",
            sources=(
                {
                    "documentId": "espocrm:contact-1",
                    "documentTitle": "EspoCRM Contact",
                    "section": "Noah Weber",
                    "snippet": "Customer: Noah Weber; Email: noah@example.test.",
                },
                {
                    "documentId": "espocrm:claim-1",
                    "documentTitle": "EspoCRM Claim",
                    "section": "TEST-CLM-2026-2501",
                    "snippet": "Claim TEST-CLM-2026-2501: Theft, status Under Review.",
                },
            ),
            tool_results=({"claim": {"id": "claim-1"}},),
        )

        selection = select_crm_context(
            result,
            "What is the status of claim TEST-CLM-2026-2501?",
        )

        contact = next(
            source
            for source in selection.sources
            if source["documentTitle"] == "EspoCRM Contact"
        )
        self.assertEqual(contact["snippet"], "Customer: Noah Weber.")
        self.assertNotIn("noah@example.test", contact["snippet"])


if __name__ == "__main__":
    unittest.main()
