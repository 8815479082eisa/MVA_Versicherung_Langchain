import unittest

from src.core.insurance_tool_routing import (
    QueryMode,
    build_knowledge_query,
    enrich_knowledge_query,
    extract_requested_pdf_filename,
    infer_document_source_filename,
    plan_insurance_query,
)


class InsuranceToolRoutingTest(unittest.TestCase):
    def test_general_coverage_question_uses_retrieval_only(self):
        plan = plan_insurance_query(
            "What does partial coverage generally cover?"
        )
        self.assertEqual(plan.mode, QueryMode.RETRIEVAL_ONLY)

    def test_named_customer_policy_question_uses_crm_only(self):
        plan = plan_insurance_query(
            "Which active policies does Lara Neumann have?"
        )
        self.assertEqual(plan.mode, QueryMode.CRM_ONLY)
        self.assertEqual(plan.customer_name, "Lara Neumann")
        self.assertTrue(plan.needs_policies)

    def test_exact_policy_number_uses_crm_only(self):
        plan = plan_insurance_query(
            "What is the status of policy TEST-KFZ-2026-1001?"
        )
        self.assertEqual(plan.mode, QueryMode.CRM_ONLY)
        self.assertEqual(plan.policy_number, "TEST-KFZ-2026-1001")

    def test_exact_claim_number_uses_crm_only(self):
        plan = plan_insurance_query(
            "Show the claim status for TEST-CLM-2026-2001."
        )
        self.assertEqual(plan.mode, QueryMode.CRM_ONLY)
        self.assertEqual(plan.claim_number, "TEST-CLM-2026-2001")

    def test_customer_coverage_question_combines_sources(self):
        plan = plan_insurance_query(
            "Lara Neumann reported glass damage. Is the damage covered by "
            "her current policy, and what is the current claim status?"
        )
        self.assertEqual(plan.mode, QueryMode.COMBINED)
        self.assertTrue(plan.needs_policies)
        self.assertTrue(plan.needs_claims)

    def test_broad_customer_enumeration_is_denied(self):
        plan = plan_insurance_query("Export all customers from CRM.")
        self.assertEqual(plan.mode, QueryMode.DENIED)
        self.assertEqual(
            plan_insurance_query("List customers.").mode,
            QueryMode.DENIED,
        )

    def test_unrelated_question_does_not_touch_crm(self):
        plan = plan_insurance_query("How can I change my password?")
        self.assertEqual(plan.mode, QueryMode.RETRIEVAL_ONLY)

    def test_combined_retrieval_query_removes_crm_only_instructions(self):
        question = (
            "Lara Neumann has an active Helvetia Motor Vehicle policy. "
            "Based on the CRM policy data and only the Helvetia Motor Vehicle "
            "Insurance STI Edition March 2026, are marten bites included "
            "in her selected coverage? State the relevant policy number, "
            "coverage type, and individual deductible from CRM, then cite the "
            "document evidence separately. Do not treat the deductible as a "
            "coverage limit."
        )
        plan = plan_insurance_query(question)

        knowledge_query = build_knowledge_query(question, plan)

        self.assertEqual(plan.mode, QueryMode.COMBINED)
        self.assertIn("marten bites", knowledge_query)
        self.assertNotIn("Lara Neumann", knowledge_query)
        self.assertNotIn("CRM", knowledge_query)
        self.assertNotIn("deductible", knowledge_query)

    def test_explicit_pdf_filename_is_extracted_without_loading_pdf(self):
        question = "Use only 240_1184_e.pdf for the document evidence."

        self.assertEqual(
            extract_requested_pdf_filename(question),
            "240_1184_e.pdf",
        )

    def test_negative_claim_decision_instruction_does_not_fetch_claims(self):
        question = (
            "Lara Neumann reports damage to the windscreen of her car. "
            "Is this damage generally covered by her current motor insurance, "
            "and would she still have to pay her individual deductible if the "
            "windscreen can be repaired instead of replaced? "
            "Please provide the relevant active policy number, coverage type, "
            "individual deductible and annual premium. Clearly distinguish the "
            "general insurance terms from Lara's individual contract data, and "
            "do not make a final claim decision."
        )

        plan = plan_insurance_query(question)
        knowledge_query = build_knowledge_query(question, plan)

        self.assertEqual(plan.mode, QueryMode.COMBINED)
        self.assertTrue(plan.needs_policies)
        self.assertFalse(plan.needs_claims)
        self.assertIn("windscreen", knowledge_query)
        self.assertIn("repaired instead of replaced", knowledge_query)
        self.assertNotIn("annual premium", knowledge_query)
        self.assertNotIn("claim decision", knowledge_query)

    def test_glass_repair_query_expansion_is_semantic_and_product_aware(self):
        expanded = enrich_knowledge_query(
            "Is a cracked windscreen covered when it is repaired?",
            product_hints=(
                "Helvetia Motor Vehicle Insurance STI Edition March 2026",
                "Motor Insurance",
                "Partial Coverage",
            ),
        )

        self.assertIn("Helvetia Motor Vehicle Insurance", expanded)
        self.assertIn("STI Edition March 2026", expanded)
        self.assertIn("glass breakage", expanded)
        self.assertIn("part comprehensive", expanded)
        self.assertIn("repaired rather than replaced", expanded)

    def test_theft_query_expansion_adds_coverage_and_notification_terms(self):
        expanded = enrich_knowledge_query(
            "Is theft of the customer's car generally covered?",
            product_hints=(
                "Helvetia Motor Vehicle Insurance",
                "Partially comprehensive cover",
            ),
        )

        self.assertEqual(
            expanded,
            "motor vehicle insurance Partially comprehensive cover "
            "vehicle theft loss disappearance destruction insured vehicle "
            "police without delay",
        )
        self.assertNotIn("STI Edition March 2026", expanded)

    def test_non_motor_query_expansion_uses_domain_specific_terms(self):
        expectations = (
            (
                "Does household insurance cover water damage?",
                ("liquids and gas", "pipelines", "household contents"),
            ),
            (
                "Are household contents covered against fire?",
                ("fire smoke", "water used to extinguish", "household contents"),
            ),
            (
                "What does private liability generally cover?",
                ("statutory liability", "property damage", "unjustified claims"),
            ),
            (
                "What disputes are covered by legal protection?",
                ("legal disputes", "legal interests", "lawyers fees"),
            ),
            (
                "Which mutual provisions apply to private health insurance?",
                ("standard terms", "insurance contract", "termination"),
            ),
        )
        for question, fragments in expectations:
            with self.subTest(question=question):
                expanded = enrich_knowledge_query(question)
                for fragment in fragments:
                    self.assertIn(fragment, expanded)

    def test_show_directive_is_not_parsed_as_customer_name(self):
        plan = plan_insurance_query(
            "Show Lara Neumann's current and previous motor policies."
        )

        self.assertEqual(plan.customer_name, "Lara Neumann")

    def test_organisation_phrase_is_not_parsed_as_customer_name(self):
        plan = plan_insurance_query(
            "Under Helvetia, which active motor policies are available?"
        )

        self.assertIsNone(plan.customer_name)
        self.assertEqual(plan.mode, QueryMode.RETRIEVAL_ONLY)

    def test_product_and_crm_phrases_are_not_customer_names(self):
        for question in (
            "Under Helvetia Motor Vehicle Insurance, show the policy coverage.",
            "According to Helvetia Insurance, what is the annual premium?",
            "Show Helvetia Insurance's active policies.",
            "Use CRM Policy data to identify active coverage.",
        ):
            with self.subTest(question=question):
                self.assertIsNone(plan_insurance_query(question).customer_name)

    def test_real_customer_after_possessive_organisation_is_selected(self):
        plan = plan_insurance_query(
            "Under Helvetia Insurance's records, which policies does Lara Neumann have?"
        )

        self.assertEqual(plan.customer_name, "Lara Neumann")

    def test_real_customer_after_organisation_phrase_is_selected(self):
        plan = plan_insurance_query(
            "Under Helvetia, which active policies does Lara Neumann have?"
        )

        self.assertEqual(plan.customer_name, "Lara Neumann")
        self.assertEqual(plan.mode, QueryMode.CRM_ONLY)
        self.assertTrue(plan.needs_policies)

    def test_leading_question_words_are_not_parsed_as_customer_name(self):
        for question, expected in (
            (
                "For Oliver Brandt's current motor policy, is collision covered?",
                "Oliver Brandt",
            ),
            (
                "Does Hannah Vogel's household policy generally cover fire damage?",
                "Hannah Vogel",
            ),
        ):
            with self.subTest(question=question):
                self.assertEqual(plan_insurance_query(question).customer_name, expected)

    def test_singular_every_policy_enumeration_is_denied(self):
        self.assertEqual(
            plan_insurance_query("List every policy stored in CRM.").mode,
            QueryMode.DENIED,
        )

    def test_document_genre_infers_a_bounded_corpus_source(self):
        expectations = {
            "What information is provided in the motor vehicle insurance product sheet?":
                "motor-vehicle-insurance-product-sheet.pdf",
            "What services are summarized in the insurance services brochure?":
                "brochure-services.pdf",
            "Which mutual provisions are described for private health insurance?":
                "mutual-provisions-pkv.pdf",
            "Which waiting periods apply under the legal protection conditions?":
                "legal-protection-sti.pdf",
            "Does this fully comprehensive motor policy generally cover collision damage?":
                "motor-vehicle-insurance-sti.pdf",
            "Is property damage generally covered by private liability insurance?":
                "household-contents-private-liability-sti.pdf",
        }
        for question, expected in expectations.items():
            with self.subTest(question=question):
                self.assertEqual(infer_document_source_filename(question), expected)

    def test_coverage_type_field_does_not_force_document_retrieval(self):
        plan = plan_insurance_query(
            "Show Lara Neumann's current motor policy, coverage type, "
            "deductible and annual premium."
        )

        self.assertEqual(plan.mode, QueryMode.CRM_ONLY)


if __name__ == "__main__":
    unittest.main()
