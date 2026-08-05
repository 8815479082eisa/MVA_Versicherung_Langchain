import csv
import unittest
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "crm"


def _rows(name):
    with (DATA_ROOT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class CRMSyntheticDataTest(unittest.TestCase):
    def test_dataset_size_natural_keys_and_reserved_emails(self):
        contacts = _rows("contacts.csv")
        policies = _rows("policies.csv")
        claims = _rows("claims.csv")
        self.assertGreaterEqual(len(contacts), 8)
        self.assertLessEqual(len(contacts), 12)
        self.assertGreaterEqual(len(policies), 12)
        self.assertLessEqual(len(policies), 20)
        self.assertGreaterEqual(len(claims), 6)
        self.assertLessEqual(len(claims), 10)
        self.assertEqual(len({row["emailAddress"] for row in contacts}), len(contacts))
        self.assertEqual(len({row["policyNumber"] for row in policies}), len(policies))
        self.assertEqual(len({row["claimNumber"] for row in claims}), len(claims))
        self.assertTrue(
            all(row["emailAddress"].endswith("@example.test") for row in contacts)
        )

    def test_lara_neumann_combined_scenario_is_linked(self):
        contacts = _rows("contacts.csv")
        policies = _rows("policies.csv")
        claims = _rows("claims.csv")
        lara_email = "lara.neumann@example.test"
        self.assertTrue(
            any(
                row["firstName"] == "Lara"
                and row["lastName"] == "Neumann"
                and row["emailAddress"] == lara_email
                for row in contacts
            )
        )
        self.assertTrue(
            any(
                row["policyNumber"] == "TEST-KFZ-2026-1001"
                and row["customerEmail"] == lara_email
                and row["productType"] == "Helvetia Motor Vehicle Insurance"
                and row["coverageType"] == "Partially comprehensive cover"
                and row["deductible"] == "150"
                for row in policies
            )
        )
        self.assertTrue(
            any(
                row["name"]
                == (
                    "Lara Neumann Helvetia Motor Vehicle Insurance Partially "
                    "comprehensive cover - STI Edition March 2026"
                )
                and row["policyNumber"] == "TEST-KFZ-2026-1003"
                and row["customerEmail"] == lara_email
                and row["productType"] == "Helvetia Motor Vehicle Insurance"
                and row["coverageType"] == "Partially comprehensive cover"
                and row["status"] == "Active"
                and row["deductible"] == "300"
                and row["annualPremium"] == "720"
                and row["currency"] == "EUR"
                for row in policies
            )
        )
        self.assertTrue(
            any(
                row["customerEmail"] == lara_email
                and row["policyNumber"] == "TEST-KFZ-2026-1001"
                and row["damageType"] == "Glass Damage"
                and row["status"] in {"Open", "Under Review"}
                for row in claims
            )
        )


if __name__ == "__main__":
    unittest.main()
