import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from src.api import rag_service


class RetrievalServiceTest(unittest.TestCase):
    def setUp(self):
        self.previous_service = rag_service._retrieval_service
        self.previous_pipeline = rag_service._pipeline
        rag_service._retrieval_service = None
        rag_service._pipeline = None

    def tearDown(self):
        rag_service._retrieval_service = self.previous_service
        rag_service._pipeline = self.previous_pipeline

    def test_pdf_hash_key_is_portable_between_windows_and_docker_paths(self):
        windows_path = r"C:\Users\mirae\project\data\raw\pdfs\policy.pdf"
        docker_path = "/app/data/raw/pdfs/policy.pdf"

        self.assertEqual(rag_service._normalize_pdf_key(windows_path), "policy.pdf")
        self.assertEqual(rag_service._normalize_pdf_key(docker_path), "policy.pdf")

    def test_retrieval_only_initialization_skips_llm_components_and_is_reused(self):
        chunks = [Document(page_content="Policy coverage", metadata={"source": "policy.pdf"})]

        def hybrid_retriever(_query: str, k: int = 5):
            del k
            return chunks

        with patch.dict(
            "os.environ",
            {"USE_INSURANCEQA_DATA": "false", "INSURANCEQA_RETRIEVAL_MODE": "off"},
        ), patch.object(
            rag_service, "get_source_files", return_value=["policy.pdf"]
        ), patch.object(
            rag_service, "load_indexed_documents_from_chroma", return_value=chunks
        ), patch.object(
            rag_service, "load_and_split_documents"
        ) as source_loader_mock, patch.object(
            rag_service, "initialize_embeddings", return_value=object()
        ), patch.object(
            rag_service, "embedding_model_has_changed", return_value=False
        ), patch.object(
            rag_service,
            "resolve_local_huggingface_model",
            side_effect=lambda model_name: model_name,
        ), patch.object(
            rag_service, "build_vectorstore", return_value=object()
        ) as build_vectorstore_mock, patch.object(
            rag_service, "build_retriever", return_value=hybrid_retriever
        ), patch.object(
            rag_service, "build_reranker", return_value=None
        ), patch.object(
            rag_service, "initialize_llm"
        ) as answer_llm_mock, patch.object(
            rag_service, "initialize_router_llm"
        ) as router_llm_mock, patch.object(
            rag_service, "initialize_query_rewrite_llm"
        ) as rewrite_llm_mock, patch.object(
            rag_service, "initialize_self_check_llm"
        ) as self_check_llm_mock, patch.object(
            rag_service, "initialize_compressor"
        ) as compressor_llm_mock, patch.object(
            rag_service, "build_compressor"
        ) as compressor_builder_mock, patch.object(
            rag_service, "build_generation_chain"
        ) as generation_chain_mock:
            first_service, first_reused = rag_service.initialize_retrieval_service(
                allow_reindex=False
            )
            second_service, second_reused = rag_service.initialize_retrieval_service(
                allow_reindex=False
            )

        self.assertIs(first_service, second_service)
        self.assertFalse(first_reused)
        self.assertTrue(second_reused)
        self.assertIsNone(rag_service._pipeline)
        build_vectorstore_mock.assert_called_once()
        self.assertFalse(build_vectorstore_mock.call_args.kwargs["allow_reindex"])
        source_loader_mock.assert_not_called()
        answer_llm_mock.assert_not_called()
        router_llm_mock.assert_not_called()
        rewrite_llm_mock.assert_not_called()
        self_check_llm_mock.assert_not_called()
        compressor_llm_mock.assert_not_called()
        compressor_builder_mock.assert_not_called()
        generation_chain_mock.assert_not_called()

    def test_failed_initialization_is_not_cached(self):
        with patch.object(
            rag_service.RetrievalService,
            "initialize",
            side_effect=RuntimeError("initialization failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                rag_service.initialize_retrieval_service(allow_reindex=False)

        self.assertIsNone(rag_service._retrieval_service)

    def test_pipeline_initialization_skips_disabled_auxiliary_components(self):
        settings = SimpleNamespace(
            retrieval=SimpleNamespace(
                enable_context_compression=False,
                force_retrieval=True,
                self_check_enabled=False,
                query_rewrite_enabled=False,
            )
        )
        pipeline = rag_service.RAGPipeline(settings)
        retrieval_service = SimpleNamespace(components={"hybrid_retriever": object()})

        with patch.object(
            rag_service, "build_answer_model_warning", return_value=None
        ), patch.object(
            rag_service,
            "initialize_retrieval_service",
            return_value=(retrieval_service, True),
        ), patch.object(
            rag_service, "initialize_llm", return_value=object()
        ), patch.object(
            pipeline, "build_compressor"
        ) as compressor_mock, patch.object(
            rag_service, "initialize_router_llm"
        ) as router_mock, patch.object(
            rag_service, "initialize_self_check_llm"
        ) as self_check_mock, patch.object(
            rag_service, "initialize_query_rewrite_llm"
        ) as rewrite_mock, patch.object(
            pipeline, "build_generation_chain"
        ) as generation_chain_mock, patch.object(
            rag_service, "_save_model_config"
        ):
            components = pipeline.initialize()

        compressor_mock.assert_not_called()
        router_mock.assert_not_called()
        self_check_mock.assert_not_called()
        rewrite_mock.assert_not_called()
        generation_chain_mock.assert_not_called()
        self.assertIsNone(components["compressor_llm"])
        self.assertIsNone(components["router_llm"])
        self.assertIsNone(components["self_check_llm"])
        self.assertIsNone(components["query_rewrite_llm"])
        self.assertNotIn("generation_chain", components)

    def test_shared_search_kwargs_are_isolated_between_concurrent_calls(self):
        service = rag_service.RetrievalService(rag_service.SETTINGS)

        class MutableRetriever:
            def __init__(self):
                self.search_kwargs = {"k": 8}

            def invoke(self, query: str):
                observed_k = self.search_kwargs["k"]
                time.sleep(0.02)
                return [Document(page_content=query, metadata={"observed_k": observed_k})]

        retriever = MutableRetriever()
        results = {}

        def invoke(query: str, top_k: int):
            results[query] = service._invoke_retriever_with_k(retriever, query, top_k)

        first = threading.Thread(target=invoke, args=("first", 2))
        second = threading.Thread(target=invoke, args=("second", 7))
        first.start()
        second.start()
        first.join()
        second.join()

        self.assertEqual(results["first"][0].metadata["observed_k"], 2)
        self.assertEqual(results["second"][0].metadata["observed_k"], 7)
        self.assertEqual(retriever.search_kwargs["k"], 8)

    def test_balanced_merge_preserves_fifth_semantic_candidate(self):
        bm25_docs = [
            Document(
                page_content=f"lexical {index}",
                metadata={"source": f"lexical-{index}.pdf", "page": index},
            )
            for index in range(1, 6)
        ]
        vector_docs = [
            Document(
                page_content=f"semantic {index}",
                metadata={"source": f"semantic-{index}.pdf", "page": index},
            )
            for index in range(1, 6)
        ]

        merged, _ = rag_service._balanced_hybrid_merge(
            bm25_docs,
            vector_docs,
            target_k=8,
        )

        merged_sources = {doc.metadata["source"] for doc in merged}
        self.assertEqual(len(merged), 8)
        self.assertIn("semantic-5.pdf", merged_sources)
        self.assertGreaterEqual(
            sum(source.startswith("semantic-") for source in merged_sources),
            5,
        )
        self.assertGreaterEqual(
            sum(source.startswith("lexical-") for source in merged_sources),
            3,
        )

    def test_pdf_citation_uses_filename_and_physical_page(self):
        document = Document(
            page_content=(
                "The deductible will not be applied if a damaged windscreen "
                "is repaired rather than replaced."
            ),
            metadata={
                "source": r"data\raw\pdfs\motor-vehicle-insurance-sti.pdf",
                "source_type": "pdf",
                "page": 8,
            },
        )

        answer = rag_service._ensure_inline_citations(
            "The repair can waive the deductible. [Doc-ID:page]",
            [document],
        )
        source = rag_service.document_to_source(document)

        self.assertIn("[motor-vehicle-insurance-sti.pdf, page 9]", answer)
        self.assertNotIn("Doc-ID", answer)
        self.assertNotIn(r"data\raw", answer)
        self.assertEqual(source.document_title, "motor-vehicle-insurance-sti.pdf")
        self.assertEqual(source.page, 9)

    def test_generation_context_has_no_automatic_qualifier_markers(self):
        document = Document(
            page_content=(
                "The deductible is waived for repair. These benefits are only "
                "provided if customer service is notified and a partner "
                "arranges the service, provided that no mainte -\n"
                "nance contract exists."
            ),
            metadata={
                "source": "motor-conditions.pdf",
                "source_type": "pdf",
                "page": 6,
            },
        )

        context = rag_service._format_context_with_sources([document])

        self.assertIn("[motor-conditions.pdf, page 7]", context)
        self.assertIn("only provided if customer service is notified", context)
        self.assertIn("provided that no mainte -\nnance contract exists", context)
        self.assertNotIn("MATERIAL QUALIFIERS PRESENT", context)
        self.assertNotIn("RELEVANT QUALIFIER SOURCE EXTRACT", context)
        self.assertNotIn("FINAL PROVIDED-THAT CONDITION", context)

    def test_generate_answer_preserves_theft_conditions_without_appending(self):
        raw_answer = (
            "General terms: Theft of the insured vehicle is covered, but no "
            "compensation is paid if the act was committed by family members "
            "[motor-vehicle-insurance-sti.pdf, page 13]. The responsible police "
            "must be notified without delay, and the insurer must be informed "
            "without delay if the vehicle is found or its whereabouts become known "
            "[motor-vehicle-insurance-sti.pdf, page 17]. Individual CRM data: policy "
            "TEST-KFZ-2026-1701 has Partially comprehensive cover, a 300 EUR "
            "deductible, and a 735 EUR annual premium "
            "[CRM: TEST-KFZ-2026-1701]. This is not a final claim decision."
        )
        docs = [
            Document(
                page_content=(
                    "The insurance does not cover damage by scorching unless caused "
                    "by a fire. K2.1.4 Theft. No compensation is paid if the act was "
                    "committed by family members."
                ),
                metadata={
                    "source": "motor-vehicle-insurance-sti.pdf",
                    "source_type": "pdf",
                    "page": 12,
                },
            ),
            Document(
                page_content=(
                    "In the case of theft you must notify the responsible police "
                    "without delay. If the vehicle is found or its whereabouts "
                    "become known, we must be informed without delay."
                ),
                metadata={
                    "source": "motor-vehicle-insurance-sti.pdf",
                    "source_type": "pdf",
                    "page": 16,
                },
            ),
            Document(
                page_content=(
                    "CRM FACT\nPolicy TEST-KFZ-2026-1701: Partially comprehensive "
                    "cover; deductible 300 EUR; annual premium 735 EUR."
                ),
                metadata={
                    "source": "espocrm:policy",
                    "source_type": "crm",
                    "section": "TEST-KFZ-2026-1701",
                },
            ),
        ]
        fake_chain = SimpleNamespace(
            invoke=lambda _: SimpleNamespace(content=raw_answer)
        )

        with patch.object(
            rag_service, "build_generation_chain", return_value=fake_chain
        ), patch.object(
            rag_service,
            "invoke_llm_stage",
            side_effect=lambda _stage, operation, **_kwargs: operation(),
        ):
            answer = rag_service.generate_answer(object(), "theft query", docs)

        self.assertEqual(answer, raw_answer)
        self.assertIn("responsible police must be notified without delay", answer)
        self.assertIn("family members", answer)
        self.assertIn("vehicle is found", answer)
        self.assertIn("[motor-vehicle-insurance-sti.pdf, page 13]", answer)
        self.assertIn("[CRM: TEST-KFZ-2026-1701]", answer)
        self.assertNotIn("unless caused by a fire", answer)
        self.assertNotIn("Material documented conditions", answer)

    def test_adjacent_context_expansion_uses_query_relevance_not_fixed_ids(self):
        anchor = Document(
            page_content="The deductible is waived for a windscreen repair.",
            metadata={
                "source": "motor-conditions.pdf",
                "page": 6,
                "start_index": 100,
            },
        )
        relevant_neighbor = Document(
            page_content=(
                "Part comprehensive insurance covers glass breakage of the "
                "windscreen."
            ),
            metadata={
                "source": "motor-conditions.pdf",
                "page": 5,
                "start_index": 800,
                "chunk_id": "neighbor-relevant",
            },
        )
        irrelevant_neighbor = Document(
            page_content="General contract administration and payment dates.",
            metadata={
                "source": "motor-conditions.pdf",
                "page": 5,
                "start_index": 0,
                "chunk_id": "neighbor-irrelevant",
            },
        )

        expanded, evidence = rag_service._expand_adjacent_source_context(
            "Is windscreen glass breakage covered by part comprehensive?",
            [anchor],
            [anchor, irrelevant_neighbor, relevant_neighbor],
            max_anchors=1,
            max_additions=1,
        )

        self.assertEqual(len(expanded), 2)
        self.assertEqual(expanded[1].metadata["chunk_id"], "neighbor-relevant")
        self.assertEqual(
            evidence[0]["added"]["chunkId"],
            "neighbor-relevant",
        )

    def test_reranker_preserves_strong_context_neighbors_of_top_anchor(self):
        documents = [
            Document(
                page_content="Top waiver rule",
                metadata={"chunk_id": "anchor"},
            ),
            Document(
                page_content="Part comprehensive heading",
                metadata={
                    "chunk_id": "heading",
                    "context_expansion_anchor_id": "anchor",
                    "context_expansion_score": 18.0,
                },
            ),
            Document(
                page_content="Windscreen breakage rule",
                metadata={
                    "chunk_id": "glass",
                    "context_expansion_anchor_id": "anchor",
                    "context_expansion_score": 27.0,
                },
            ),
            Document(
                page_content="Other high reranker result",
                metadata={"chunk_id": "other-1"},
            ),
            Document(
                page_content="Other result",
                metadata={"chunk_id": "other-2"},
            ),
        ]

        selected = rag_service._preserve_high_value_context_neighbors(
            documents,
            [0, 3, 4],
            top_k=5,
        )

        self.assertEqual(selected[:3], [0, 2, 1])

    def test_reranker_prefers_matching_product_domain_over_higher_generic_score(self):
        documents = [
            Document(
                page_content="Part comprehensive motor cover includes vehicle glass.",
                metadata={"source": "motor-vehicle-terms.pdf"},
            ),
            Document(
                page_content="Comprehensive general insurance wording.",
                metadata={"source": "household-contents-terms.pdf"},
            ),
        ]

        class Reranker:
            def compute_score(self, _pairs):
                return [0.2, 0.95]

        selected = rag_service.rerank_documents(
            "Is windscreen damage covered by my motor insurance?",
            documents,
            Reranker(),
            top_k=1,
        )

        self.assertEqual(selected[0].metadata["source"], "motor-vehicle-terms.pdf")

    def test_reranker_does_not_score_product_incompatible_candidates_when_capacity_remains(self):
        documents = [
            Document(
                page_content="The vehicle theft provision covers loss of the insured car.",
                metadata={"source": "motor-vehicle-terms.pdf"},
            ),
            Document(
                page_content="Household contents are covered against burglary.",
                metadata={"source": "household-contents-terms.pdf"},
            ),
        ]

        class Reranker:
            pair_count = 0

            def compute_score(self, pairs):
                self.pair_count = len(pairs)
                return [0.8] * len(pairs)

        reranker = Reranker()
        selected = rag_service.rerank_documents(
            "Is theft of the customer's car covered by motor insurance?",
            documents,
            reranker,
            top_k=1,
        )

        self.assertEqual(reranker.pair_count, 1)
        self.assertEqual(selected[0].metadata["source"], "motor-vehicle-terms.pdf")

    def test_reranker_keeps_query_evidence_ahead_of_generic_neighbor_score(self):
        documents = [
            Document(
                page_content=(
                    "17/31 STI Helvetia Motor Vehicle Insurance. "
                    "Bonus levels and compensation values are described here."
                ),
                metadata={
                    "source": "motor-vehicle-insurance-sti.pdf",
                    "page": 16,
                    "chunk_id": "generic-neighbor",
                },
            ),
            Document(
                page_content=(
                    "K2.1.5 Glass. The glass insurance covers involuntary breakage "
                    "as well as glass damage caused by accidents to the front and rear "
                    "windscreens, the side windows and the sunroof, which make it "
                    "necessary to replace or repair the glass for safety reasons."
                ),
                metadata={
                    "source": "motor-vehicle-insurance-sti.pdf",
                    "page": 13,
                    "chunk_id": "glass-cover",
                },
            ),
            Document(
                page_content=(
                    "G10.2 You will not have to bear a deductible: "
                    "g) if the damaged front windscreen is repaired and not replaced "
                    "in the case of glass damage."
                ),
                metadata={
                    "source": "motor-vehicle-insurance-sti.pdf",
                    "page": 8,
                    "chunk_id": "deductible-waiver",
                },
            ),
        ]

        class Reranker:
            def compute_score(self, _pairs):
                return [0.99, 0.2, 0.1]

        selected = rag_service.rerank_documents(
            "Is windscreen glass damage covered, and is the deductible waived when repaired?",
            documents,
            Reranker(),
            top_k=2,
        )

        self.assertEqual(
            [doc.metadata["chunk_id"] for doc in selected],
            ["deductible-waiver", "glass-cover"],
        )

    def test_reranker_prioritizes_theft_coverage_and_duties_over_collision_score(self):
        documents = [
            Document(
                page_content=(
                    "Collision events cover damage caused by sudden and violent "
                    "external effects, impact or crashing."
                ),
                metadata={"source": "motor-vehicle-terms.pdf", "chunk_id": "collision"},
            ),
            Document(
                page_content=(
                    "Theft. The insurance covers loss, disappearance, destruction "
                    "or damage caused by theft of the insured vehicle. No compensation "
                    "is paid if the act was committed by family members."
                ),
                metadata={"source": "motor-vehicle-terms.pdf", "chunk_id": "theft-cover"},
            ),
            Document(
                page_content=(
                    "Damage caused by theft must be reported to the responsible police "
                    "without delay."
                ),
                metadata={"source": "motor-vehicle-terms.pdf", "chunk_id": "theft-duty"},
            ),
        ]

        class Reranker:
            def compute_score(self, _pairs):
                return [0.99, 0.1, 0.05]

        selected = rag_service.rerank_documents(
            "Is theft of the insured car covered and which conditions apply?",
            documents,
            Reranker(),
            top_k=2,
        )

        self.assertEqual(
            [doc.metadata["chunk_id"] for doc in selected],
            ["theft-cover", "theft-duty"],
        )

    def test_product_affinity_is_not_hardcoded_to_motor_insurance(self):
        documents = [
            Document(
                page_content="Emergency treatment while travelling.",
                metadata={"source": "international-travel-policy.pdf"},
            ),
            Document(
                page_content="Veterinary treatment for a dog.",
                metadata={"source": "pet-health-policy.pdf"},
            ),
        ]

        self.assertGreater(
            rag_service._insurance_product_affinity(
                "Does my travel insurance cover this trip?", documents[0]
            ),
            rag_service._insurance_product_affinity(
                "Does my travel insurance cover this trip?", documents[1]
            ),
        )
        self.assertGreater(
            rag_service._insurance_product_affinity(
                "Does pet insurance cover my dog?", documents[1]
            ),
            rag_service._insurance_product_affinity(
                "Does pet insurance cover my dog?", documents[0]
            ),
        )

    def test_adjacent_context_expansion_includes_same_page_governing_heading(self):
        heading = Document(
            page_content="G10.2 You will not have to bear a deductible:",
            metadata={
                "source": "motor-terms.pdf",
                "page": 8,
                "start_index": 100,
                "chunk_id": "heading",
            },
        )
        anchor = Document(
            page_content=(
                "g) if the damaged front windscreen is repaired and not replaced "
                "in the case of glass damage."
            ),
            metadata={
                "source": "motor-terms.pdf",
                "page": 8,
                "start_index": 500,
                "chunk_id": "repair-rule",
            },
        )

        expanded, _ = rag_service._expand_adjacent_source_context(
            "Is the deductible charged when the windscreen is repaired?",
            [anchor],
            [heading, anchor],
            max_anchors=1,
            max_additions=1,
        )

        self.assertEqual(expanded[1].metadata["chunk_id"], "heading")
        self.assertGreaterEqual(expanded[1].metadata["context_expansion_score"], 30)

    def test_adjacent_context_expansion_considers_late_waiver_anchor(self):
        filler = [
            Document(
                page_content=f"Other motor condition {index}.",
                metadata={
                    "source": "motor-terms.pdf",
                    "page": index,
                    "start_index": 0,
                    "chunk_id": f"filler-{index}",
                },
            )
            for index in range(6)
        ]
        waiver_heading = Document(
            page_content="G10.2 You will not have to bear a deductible:",
            metadata={
                "source": "motor-terms.pdf",
                "page": 8,
                "start_index": 100,
                "chunk_id": "waiver-heading",
            },
        )
        waiver_rule = Document(
            page_content=(
                "g) if the damaged front windscreen is repaired and not replaced "
                "in the case of glass damage."
            ),
            metadata={
                "source": "motor-terms.pdf",
                "page": 8,
                "start_index": 500,
                "chunk_id": "waiver-rule",
            },
        )

        expanded, _ = rag_service._expand_adjacent_source_context(
            "Is the deductible charged when the windscreen is repaired?",
            filler + [waiver_rule],
            filler + [waiver_heading, waiver_rule],
            max_anchors=8,
            max_additions=3,
        )

        self.assertIn(
            "waiver-heading",
            [doc.metadata.get("chunk_id") for doc in expanded],
        )

    def test_expansion_and_reranker_keep_waiver_pair_across_two_chunk_boundaries(self):
        distractor = Document(
            page_content="Other deductible conditions for motor insurance.",
            metadata={
                "source": "motor-terms.pdf",
                "page": 8,
                "start_index": 100,
                "chunk_id": "distractor",
            },
        )
        heading = Document(
            page_content="G10.2 You will not have to bear a deductible:",
            metadata={
                "source": "motor-terms.pdf",
                "page": 8,
                "start_index": 500,
                "chunk_id": "waiver-heading",
            },
        )
        waiver_rule = Document(
            page_content=(
                "g) if the damaged front windscreen is repaired and not replaced "
                "in the case of glass damage."
            ),
            metadata={
                "source": "motor-terms.pdf",
                "page": 8,
                "start_index": 900,
                "chunk_id": "waiver-rule",
            },
        )
        following_page = Document(
            page_content="G10.3 A single deductible applies to towing vehicles.",
            metadata={
                "source": "motor-terms.pdf",
                "page": 9,
                "start_index": 0,
                "chunk_id": "following-page",
            },
        )

        expanded, _ = rag_service._expand_adjacent_source_context(
            "Is the deductible charged when the windscreen is repaired?",
            [following_page],
            [distractor, heading, waiver_rule, following_page],
            max_anchors=1,
            max_additions=2,
        )

        self.assertEqual(
            [doc.metadata.get("chunk_id") for doc in expanded[1:]],
            ["waiver-rule", "waiver-heading"],
        )
        selected = rag_service._preserve_high_value_context_neighbors(
            expanded,
            [0, 1],
            top_k=2,
        )
        self.assertEqual(
            [expanded[index].metadata.get("chunk_id") for index in selected],
            ["waiver-rule", "waiver-heading"],
        )

    def test_answer_guard_adds_waiver_only_with_complete_selected_evidence(self):
        docs = [
            Document(
                page_content="G10.2 You will not have to bear a deductible:",
                metadata={"source": "motor-terms.pdf", "page": 8, "start_index": 100},
            ),
            Document(
                page_content=(
                    "g) if the damaged front windscreen is repaired and not replaced "
                    "in the case of glass damage."
                ),
                metadata={"source": "motor-terms.pdf", "page": 8, "start_index": 500},
            ),
        ]

        answer = rag_service._ensure_evidence_backed_windscreen_waiver(
            "The individual deductible may still apply to a windscreen repair.",
            docs,
        )

        self.assertIn("will not have to bear a deductible", answer)
        self.assertIn("repaired and not replaced", answer)
        self.assertNotIn("may still apply", answer)

        answer_with_generic_exception = rag_service._ensure_evidence_backed_windscreen_waiver(
            "The deductible applies unless specified otherwise in the policy terms.",
            docs,
        )
        self.assertNotIn("applies unless", answer_with_generic_exception)
        self.assertIn("will not have to bear a deductible", answer_with_generic_exception)

        contradictory_repair_condition = (
            "If the windscreen is repaired rather than replaced, the insurance does "
            "not cover the cost if the repair is not carried out."
        )
        repaired_answer = rag_service._ensure_evidence_backed_windscreen_waiver(
            contradictory_repair_condition,
            docs,
        )
        self.assertNotIn("does not cover the cost", repaired_answer)
        self.assertIn("will not have to bear a deductible", repaired_answer)

        still_applies_answer = rag_service._ensure_evidence_backed_windscreen_waiver(
            (
                "If the damaged front windscreen is repaired and not replaced, "
                "the deductible still applies."
            ),
            docs,
        )
        self.assertNotIn("still applies", still_applies_answer)
        self.assertIn("will not have to bear a deductible", still_applies_answer)

        alternate_contradictions = (
            "There is no explicit mention that repair of the windscreen removes the "
            "deductible. Compensation for glass damage is provided unless the damaged "
            "front windscreen is repaired and not replaced."
        )
        corrected_alternate = rag_service._ensure_evidence_backed_windscreen_waiver(
            alternate_contradictions,
            docs,
        )
        self.assertNotIn("no explicit mention", corrected_alternate)
        self.assertNotIn("provided unless", corrected_alternate)
        self.assertIn("will not have to bear a deductible", corrected_alternate)

    def test_answer_guard_does_not_invent_waiver_without_governing_heading(self):
        docs = [
            Document(
                page_content="A damaged windscreen may be repaired rather than replaced.",
                metadata={"source": "motor-terms.pdf", "page": 8},
            )
        ]
        original = "The contract has an individual deductible of 300 EUR."

        answer = rag_service._ensure_evidence_backed_windscreen_waiver(original, docs)

        self.assertEqual(answer, original)

if __name__ == "__main__":
    unittest.main()
