import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from src.api import rag_service


class RagSmokeTest(unittest.TestCase):
    def test_run_rag_with_small_local_corpus(self):
        corpus = [
            Document(page_content="Die Deckungssumme fuer Personenschaeden betraegt 100 Mio EUR.", metadata={"source": "policy.pdf", "page": 12}),
            Document(page_content="Selbstbeteiligung bei Diebstahl: 300 EUR.", metadata={"source": "policy.pdf", "page": 20}),
        ]

        def hybrid_retriever(query: str, k: int = 8):
            del k
            q = query.lower()
            return [d for d in corpus if "deckung" in q and "deckungssumme" in d.page_content.lower()] or corpus[:1]

        components = {
            "hybrid_retriever": hybrid_retriever,
            "reranker_model": None,
            "compressor_llm": object(),
            "llm": object(),
            "router_llm": object(),
            "self_check_llm": object(),
            "query_rewrite_llm": object(),
            "generation_chain": object(),
        }

        with patch.object(rag_service.RAGPipeline, "initialize", return_value=components), patch(
            "src.api.rag_service.perform_self_check", side_effect=lambda *args, **kwargs: (args[2], args[3])
        ), patch(
            "src.api.rag_service.generate_answer", return_value="Die Deckungssumme betraegt 100 Mio EUR."
        ), patch(
            "src.api.rag_service.pdfs_have_changed", return_value=False
        ):
            rag_service._pipeline = None
            result = rag_service.run_rag("Wie hoch ist die Deckungssumme?", chat_history=[])

        self.assertIn("Deckungssumme", result.answer)
        self.assertGreaterEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].document_title, "policy.pdf")


if __name__ == "__main__":
    unittest.main()
