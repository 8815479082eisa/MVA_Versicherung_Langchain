import unittest

from src.evaluation.alce_metrics import (
    citation_precision,
    citation_recall,
    extract_citation_indices,
    split_into_statements,
)


class MockNLI:
    def entail(self, premise: str, hypothesis: str) -> bool:
        h = hypothesis.lower()
        p = premise.lower()

        if "earth is round" in h:
            return "earth is round" in p

        if "claim" in h:
            has_p1 = "unrelated evidence" in p
            has_p2 = "claim is true" in p
            if has_p1 and has_p2:
                return True
            if has_p2:
                return True
            return False

        return False


class AlceMetricsTest(unittest.TestCase):
    def setUp(self):
        self.nli = MockNLI()

    def test_split_into_statements(self):
        text = "First sentence. Second sentence!\n- Bullet claim [1]\nThird question?"
        statements = split_into_statements(text)
        self.assertEqual(len(statements), 4)
        self.assertEqual(statements[0], "First sentence.")
        self.assertEqual(statements[1], "Second sentence!")
        self.assertEqual(statements[2], "Bullet claim [1]")
        self.assertEqual(statements[3], "Third question?")

    def test_extract_citation_indices(self):
        statement = "Claim [1][2] plus repeated [2] and [10]."
        self.assertEqual(extract_citation_indices(statement), [1, 2, 10])

    def test_citation_recall_with_mock_nli(self):
        statements = ["Earth is round [1].", "Mars is blue."]
        cited = [
            [{"title": "Doc1", "text": "Earth is round."}],
            [],
        ]
        recall = citation_recall(statements, cited, self.nli)
        self.assertAlmostEqual(recall, 0.5, places=6)

    def test_citation_precision_irrelevance_logic(self):
        statements = ["Claim [1][2]."]
        cited = [
            [
                {"title": "Doc1", "text": "Unrelated evidence."},
                {"title": "Doc2", "text": "Claim is true."},
            ]
        ]

        precision = citation_precision(statements, cited, self.nli)
        self.assertAlmostEqual(precision, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
