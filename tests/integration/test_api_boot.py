import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except Exception:
    TestClient = None

if TestClient is not None:
    from src.main import app
    from src.api.rag_service import AnswerResult, Source


@unittest.skipIf(TestClient is None, "fastapi is not installed in this environment")
class ApiBootTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body.get("status"), "ok")
        self.assertIn("configured_answer_model", body)
        self.assertIn("preferred_answer_model", body)
        self.assertIn("answer_model_matches_preference", body)
        self.assertIn("query_rewrite_enabled", body)
        self.assertIn("nemo_enforce_output", body)
        self.assertIn("safety_backend", body)

    def test_ask_endpoint_schema(self):
        mocked = AnswerResult(
            answer="Mock answer",
            sources=[Source(document_id="doc1", document_title="Doc 1", page=1, section=None, snippet="x")],
            query="q",
            latency_ms=12,
        )
        with patch("src.main.rag_service.run_rag", return_value=mocked):
            response = self.client.post("/api/ask", json={"question": "Was ist versichert?"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("answer", body)
        self.assertIn("sources", body)
        self.assertIn("latencyMs", body)
        self.assertIsInstance(body["sources"], list)


if __name__ == "__main__":
    unittest.main()
