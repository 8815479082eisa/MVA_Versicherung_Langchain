import unittest
from datetime import date

from src.core.current_policy import select_current_policy_records


REFERENCE_DATE = date(2026, 8, 1)


def policy(
    number: str,
    customer: str,
    product: str,
    status: str,
    start: str | None,
    end: str | None,
) -> dict[str, str | None]:
    return {
        "entity_type": "policy",
        "policy_number": number,
        "customer_name": customer,
        "product_type": product,
        "status": status,
        "start_date": start,
        "end_date": end,
    }


class CurrentPolicySelectionTests(unittest.TestCase):
    def setUp(self):
        self.lara_old = policy(
            "TEST-KFZ-2026-1001",
            "Lara Neumann",
            "Helvetia Motor Vehicle Insurance",
            "Active",
            "2026-01-01",
            "2026-12-31",
        )
        self.lara_current = policy(
            "TEST-KFZ-2026-1003",
            "Lara Neumann",
            "Helvetia Motor Vehicle Insurance",
            "Active",
            "2026-07-01",
            "2027-06-30",
        )

    def test_a_current_lara_selects_latest_effective_policy(self):
        result = select_current_policy_records(
            "What are the coverage type, deductible and annual premium of Lara "
            "Neumann's current motor insurance policy?",
            [self.lara_old, self.lara_current],
            reference_date=REFERENCE_DATE,
        )

        self.assertTrue(result.intent_activated)
        self.assertEqual(
            [record["policy_number"] for record in result.selected_records],
            ["TEST-KFZ-2026-1003"],
        )

    def test_b_explicit_old_policy_does_not_activate_current_filter(self):
        records = [self.lara_old, self.lara_current]
        result = select_current_policy_records(
            "Show the earlier policy TEST-KFZ-2026-1001 for Lara Neumann.",
            records,
            reference_date=REFERENCE_DATE,
        )

        self.assertFalse(result.intent_activated)
        self.assertEqual(result.selected_records, tuple(records))

    def test_c_historical_query_keeps_old_policy_reachable(self):
        records = [self.lara_old, self.lara_current]
        result = select_current_policy_records(
            "Which motor insurance policy did Lara Neumann have previously?",
            records,
            reference_date=REFERENCE_DATE,
        )

        self.assertFalse(result.intent_activated)
        self.assertEqual(result.selected_records, tuple(records))

    def test_d_comparison_query_does_not_discard_older_policy(self):
        records = [self.lara_old, self.lara_current]
        result = select_current_policy_records(
            "Compare Lara Neumann's previous and current motor insurance policies.",
            records,
            reference_date=REFERENCE_DATE,
        )

        self.assertFalse(result.intent_activated)
        self.assertEqual(result.selected_records, tuple(records))

    def test_other_customer_current_policy_is_selected_generally(self):
        old = policy(
            "TEST-KFZ-2025-9001",
            "Alex Example",
            "Motor Insurance",
            "Expired",
            "2025-01-01",
            "2025-12-31",
        )
        current = policy(
            "TEST-KFZ-2026-9002",
            "Alex Example",
            "Motor Insurance",
            "Active",
            "2026-03-01",
            "2027-02-28",
        )
        result = select_current_policy_records(
            "What is Alex Example's currently active motor insurance policy?",
            [self.lara_current, old, current],
            reference_date=REFERENCE_DATE,
        )

        self.assertTrue(result.intent_activated)
        self.assertEqual(result.selected_records, (current,))

    def test_e_non_current_query_leaves_candidates_unchanged(self):
        records = [self.lara_old, self.lara_current]
        result = select_current_policy_records(
            "What deductible is shown for Lara Neumann's motor insurance?",
            records,
            reference_date=REFERENCE_DATE,
        )

        self.assertFalse(result.intent_activated)
        self.assertEqual(result.selected_records, tuple(records))

    def test_list_query_leaves_all_policy_candidates_available(self):
        records = [self.lara_old, self.lara_current]
        result = select_current_policy_records(
            "List Lara Neumann's motor insurance policies.",
            records,
            reference_date=REFERENCE_DATE,
        )

        self.assertFalse(result.intent_activated)
        self.assertEqual(result.selected_records, tuple(records))

    def test_f_missing_or_conflicting_metadata_fails_safe(self):
        missing_end = policy(
            "TEST-KFZ-2026-9998",
            "Lara Neumann",
            "Motor Insurance",
            "Active",
            "2026-07-01",
            None,
        )
        same_start = policy(
            "TEST-KFZ-2026-9999",
            "Lara Neumann",
            "Motor Insurance",
            "Active",
            "2026-07-01",
            "2027-06-30",
        )
        result = select_current_policy_records(
            "What is Lara Neumann's current motor insurance policy?",
            [missing_end, self.lara_current, same_start],
            reference_date=REFERENCE_DATE,
        )

        self.assertTrue(result.intent_activated)
        self.assertEqual(result.selected_records, ())
        self.assertIn("ambiguous", result.reason.lower())


if __name__ == "__main__":
    unittest.main()
