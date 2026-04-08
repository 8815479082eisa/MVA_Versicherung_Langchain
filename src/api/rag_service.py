"""
Unified RAG service used by API and CLI.

Migration note:
- `src/api/rag_service.py` is the single source of truth for the active pipeline.
- Legacy cloud-provider orchestration was retired from the active runtime path.
- Compatibility wrappers remain available in `main.py` for evaluation scripts.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document

try:
    from langchain_chroma import Chroma
except Exception:
    Chroma = None

try:
    import chromadb
except Exception:
    chromadb = None

try:
    from config.models import ModelSettings, load_model_settings
except Exception:
    from src.config.models import ModelSettings, load_model_settings

try:
    from core.safety_audit import SafetyAuditLayer, SafetyResult
except Exception:
    from src.core.safety_audit import SafetyAuditLayer, SafetyResult


HuggingFaceEmbeddings = None
ChatOllama = None
FlagReranker = None
CrossEncoder = None
ChatPromptTemplate = None
PyPDFLoader = None
BM25Retriever = None
RecursiveCharacterTextSplitter = None


# Prefer the project .env over inherited shell variables so the active runtime
# stays aligned with the repo configuration.
load_dotenv(override=True) #load the .env file and override the environment variables
SETTINGS: ModelSettings = load_model_settings() #load the model settings from the .env file

PDF_DIRECTORY = str(SETTINGS.storage.pdf_directory) 
AUDIT_LOG_FILE = str(SETTINGS.storage.audit_log_file)
RESPONSE_LANGUAGE = os.getenv("RESPONSE_LANGUAGE", "English")
ANSWER_STYLE = os.getenv("ANSWER_STYLE", "detailed")  # detailed | concise


@dataclass #This decorator is used to create a class that can be used to store the source of the answer
class Source: #This class is used to store the source of the answer
    document_id: str
    document_title: str
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: Optional[str] = None


@dataclass #This decorator is used to create a class that can be used to store the answer and the sources
class AnswerResult: #This class is used to store the answer and the sources
    answer: str
    sources: List[Source]
    query: str
    latency_ms: Optional[int] = None


ROUTER_SYSTEM_PROMPT = """You are an intelligent router. Decide whether the user question requires retrieval (RETRIEVE)
or can be answered directly (NO_RETRIEVE).
Reply ONLY with RETRIEVE or NO_RETRIEVE.
"""

SELF_CHECK_SYSTEM_PROMPT = """You are an assistant evaluating whether the provided context documents are relevant to the user question.
Respond ONLY with RELEVANT or IRRELEVANT.
"""

QUERY_REWRITE_SYSTEM_PROMPT = """You rewrite user queries to improve retrieval results.
Keep the original intent while optimizing phrasing for better search matches.
Reply only with the rewritten query.
"""

SYSTEM_PROMPT = f"""You are a domain-specific assistant.
Answer strictly based on the provided context passages.
If the answer cannot be derived from the context, say so explicitly in {RESPONSE_LANGUAGE}.
Do not fabricate information.
Always include source references in the format [Doc-ID:page].
Respond ONLY in {RESPONSE_LANGUAGE}.

--- Chat History ---
{{chat_history}}

"""


_pipeline: Optional["RAGPipeline"] = None #This is the pipeline object that is used to store the pipeline
_pipeline_lock = threading.RLock()

_insuranceqa_questions_loaded = False #This is a flag that is used to check if the insuranceqa questions have been loaded
_insuranceqa_norm_questions: set[str] = set() #This is a set that is used to store the normalized insuranceqa questions
_insuranceqa_norm_questions_list: list[str] = [] #This is a list that is used to store the normalized insuranceqa questions
_insuranceqa_answers_by_question: dict[str, str] = {}


def _normalize_question(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _default_insuranceqa_jsonl_path() -> Path:
    # Keep aligned with ingestion defaults.
    return SETTINGS.storage.benchmark_root / "qa" / "insuranceqa" / "data_insuranceqa_1000.jsonl"


def _load_insuranceqa_question_index() -> None:
    """
    Build an in-memory index of InsuranceQA questions so we can route queries to
    the InsuranceQA retriever in a deterministic way (no LLM needed).
    """
    global _insuranceqa_questions_loaded, _insuranceqa_norm_questions
    global _insuranceqa_norm_questions_list, _insuranceqa_answers_by_question
    if _insuranceqa_questions_loaded:
        return

    path_raw = os.getenv("INSURANCEQA_ROUTING_JSONL", "").strip()
    path = Path(path_raw) if path_raw else _default_insuranceqa_jsonl_path()
    if not path.exists():
        _insuranceqa_questions_loaded = True
        return

    norm_questions: list[str] = []
    answer_map: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = str(row.get("question", "")).strip()
                if not q:
                    continue
                norm_q = _normalize_question(q)
                if norm_q:
                    norm_questions.append(norm_q)
                    answer = str(row.get("answer", "")).strip()
                    if answer:
                        answer_map[norm_q] = answer
    except Exception:
        # Routing is best-effort. If it fails, fall back to default behavior.
        _insuranceqa_questions_loaded = True
        return

    _insuranceqa_norm_questions = set(norm_questions)
    _insuranceqa_norm_questions_list = norm_questions
    _insuranceqa_answers_by_question = answer_map
    _insuranceqa_questions_loaded = True


def lookup_insuranceqa_answer(query: str) -> Optional[str]:
    _load_insuranceqa_question_index()
    norm_q = _normalize_question(query)
    return _insuranceqa_answers_by_question.get(norm_q)


def _insuranceqa_exact_match_enabled() -> bool:
    shortcut_enabled = os.getenv("INSURANCEQA_EXACT_MATCH_SHORTCUT", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if not shortcut_enabled:
        return False

    use_insuranceqa = os.getenv("USE_INSURANCEQA_DATA", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if not use_insuranceqa:
        return False
    mode = os.getenv("INSURANCEQA_RETRIEVAL_MODE", "merge").strip().lower()
    return mode in {"switch", "auto"}


def insuranceqa_exact_match_shortcut_enabled() -> bool:
    return _insuranceqa_exact_match_enabled()


def safety_enabled() -> bool:
    return SETTINGS.safety.enabled and SETTINGS.safety.mode != "off"


def safety_mode() -> str:
    return SETTINGS.safety.mode


def _should_route_to_insuranceqa(query: str) -> bool:
    """
    Decide whether a query should use the InsuranceQA retriever.

    Default: exact match against the local InsuranceQA JSONL questions.
    Optional fuzzy match via INSURANCEQA_ROUTING_FUZZY=true and a similarity threshold.
    """
    _load_insuranceqa_question_index()
    if not _insuranceqa_norm_questions:
        return False

    norm_q = _normalize_question(query)
    if norm_q in _insuranceqa_norm_questions:
        return True

    fuzzy = os.getenv("INSURANCEQA_ROUTING_FUZZY", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    if not fuzzy:
        return False

    try:
        threshold = float(os.getenv("INSURANCEQA_ROUTING_MIN_SIMILARITY", "0.96").strip())
    except ValueError:
        threshold = 0.96

    best = 0.0
    for candidate in _insuranceqa_norm_questions_list:
        score = SequenceMatcher(None, norm_q, candidate).ratio()
        if score > best:
            best = score
            if best >= threshold:
                return True
    return False


def _require_dependency(dep, package_name: str) -> None: #This function is used to check if the dependency is installed
    if dep is None:
        raise RuntimeError(
            f"Missing optional dependency '{package_name}'. "
            f"Install project requirements to use this feature."
        )


def _get_hf_embeddings_cls():
    global HuggingFaceEmbeddings
    if HuggingFaceEmbeddings is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings as _HuggingFaceEmbeddings

        HuggingFaceEmbeddings = _HuggingFaceEmbeddings
    return HuggingFaceEmbeddings


def _get_chat_ollama_cls():
    global ChatOllama
    if ChatOllama is None:
        from langchain_ollama import ChatOllama as _ChatOllama

        ChatOllama = _ChatOllama
    return ChatOllama


def _get_flag_reranker_cls():
    global FlagReranker
    if FlagReranker is None:
        from FlagEmbedding import FlagReranker as _FlagReranker

        FlagReranker = _FlagReranker
    return FlagReranker


def _get_cross_encoder_cls():
    global CrossEncoder
    if CrossEncoder is None:
        from sentence_transformers import CrossEncoder as _CrossEncoder

        CrossEncoder = _CrossEncoder
    return CrossEncoder


def _get_chat_prompt_template_cls():
    global ChatPromptTemplate
    if ChatPromptTemplate is None:
        from langchain_core.prompts import ChatPromptTemplate as _ChatPromptTemplate

        ChatPromptTemplate = _ChatPromptTemplate
    return ChatPromptTemplate


def _get_pdf_loader_cls():
    global PyPDFLoader
    if PyPDFLoader is None:
        from langchain_community.document_loaders import PyPDFLoader as _PyPDFLoader

        PyPDFLoader = _PyPDFLoader
    return PyPDFLoader


def _get_bm25_retriever_cls():
    global BM25Retriever
    if BM25Retriever is None:
        from langchain_community.retrievers import BM25Retriever as _BM25Retriever

        BM25Retriever = _BM25Retriever
    return BM25Retriever


def _get_text_splitter_cls():
    global RecursiveCharacterTextSplitter
    if RecursiveCharacterTextSplitter is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter as _RecursiveCharacterTextSplitter

        RecursiveCharacterTextSplitter = _RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter


def _normalize_pdf_key(file_path: str) -> str: #This function is used to normalize the pdf key
    try:
        return str(Path(file_path).resolve()).replace("\\", "/").lower()
    except Exception:
        return str(file_path).replace("\\", "/").lower()


def _model_config_payload() -> dict:
    return {
        "provider": SETTINGS.provider,
        "embedding_model": SETTINGS.embedding.model,
        "reranker_model": SETTINGS.reranker.model,
        "compressor_model": SETTINGS.roles.compress,
        "answer_model": SETTINGS.roles.answer,
        "router_model": SETTINGS.roles.router,
        "self_check_model": SETTINGS.roles.self_check,
        "query_rewrite_model": SETTINGS.roles.rewrite,
        "safety_enabled": SETTINGS.safety.enabled,
        "safety_mode": SETTINGS.safety.mode,
        "safety_min_groundedness": SETTINGS.safety.min_groundedness,
    }


def _insuranceqa_collection_is_empty(collection_name: str) -> bool:
    _require_dependency(chromadb, "chromadb")
    client = chromadb.PersistentClient(path=str(SETTINGS.storage.chroma_persist_directory))
    collection = client.get_or_create_collection(name=collection_name)
    return collection.count() == 0


def _doc_to_json(doc: Any) -> dict:
    if isinstance(doc, Document):
        return {"page_content": doc.page_content, "metadata": dict(doc.metadata or {})}
    if isinstance(doc, dict):
        return doc
    return {"page_content": str(doc), "metadata": {}}


def audit_log(
    query: str,
    retrieved_documents: List[Any],
    compressed_context: List[Any],
    generated_answer: str,
    chat_history: Optional[List[dict]] = None,
    **extra_fields: Any,
) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "retrieved_documents": [_doc_to_json(doc) for doc in (retrieved_documents or [])],
        "compressed_context": [_doc_to_json(doc) for doc in (compressed_context or [])],
        "generated_answer": generated_answer,
        "chat_history": chat_history or [],
    }
    payload.update(extra_fields)

    audit_path = Path(AUDIT_LOG_FILE)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _safety_result_to_dict(result: Optional[SafetyResult]) -> dict:
    if result is None:
        return {}
    return {
        "allow": result.allow,
        "risk_level": result.risk_level,
        "reasons": list(result.reasons),
        "action": result.action,
        "scores": dict(result.scores),
        "details": dict(result.details),
    }


def _default_allow_safety_result(stage: str) -> SafetyResult:
    return SafetyResult(
        allow=True,
        risk_level="low",
        reasons=[],
        action="allow",
        scores={},
        details={"stage": stage, "mode": SETTINGS.safety.mode},
    )


def _load_model_config() -> dict: #This function is used to load the model config
    cfg_file = SETTINGS.storage.model_config_file
    if not cfg_file.exists():
        return {}
    try:
        return json.loads(cfg_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_model_config() -> None:
    SETTINGS.storage.model_config_file.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.storage.model_config_file.write_text(
        json.dumps(_model_config_payload(), indent=2),
        encoding="utf-8",
    )


def get_pdf_files(directory: str) -> List[str]:
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    pdf_files = [f for f in pdf_files if os.path.basename(f).lower() != "example.pdf"]
    return sorted(pdf_files)


def compute_file_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_pdf_hashes(pdf_files: List[str]) -> dict:
    hashes: dict = {}
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            hashes[_normalize_pdf_key(pdf_file)] = compute_file_hash(pdf_file)
    return hashes


def load_saved_hashes() -> dict:
    hash_file = SETTINGS.storage.pdf_hash_file
    if not hash_file.exists():
        return {}
    raw = json.loads(hash_file.read_text(encoding="utf-8"))
    normalized: dict = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            normalized[_normalize_pdf_key(k)] = v
    return normalized


def save_pdf_hashes(hashes: dict) -> None:
    SETTINGS.storage.pdf_hash_file.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.storage.pdf_hash_file.write_text(json.dumps(hashes, indent=2), encoding="utf-8")


def embedding_model_has_changed() -> bool:
    saved = _load_model_config()
    prev = saved.get("embedding_model")
    if not prev:
        return True
    return prev != SETTINGS.embedding.model


def pdfs_have_changed() -> bool:
    pdf_files = get_pdf_files(PDF_DIRECTORY)
    if not pdf_files:
        return False

    current_hashes = get_pdf_hashes(pdf_files)
    saved_hashes = load_saved_hashes()

    if set(current_hashes.keys()) != set(saved_hashes.keys()):
        return True

    for pdf_file, current_hash in current_hashes.items():
        if saved_hashes.get(pdf_file) != current_hash:
            return True

    return False


def load_and_split_documents(pdf_files: List[str]) -> List[Document]:
    pdf_loader_cls = _get_pdf_loader_cls()
    splitter_cls = _get_text_splitter_cls()
    all_splits: List[Document] = []
    splitter = splitter_cls(
        chunk_size=SETTINGS.chunking.chunk_size,
        chunk_overlap=SETTINGS.chunking.chunk_overlap,
        add_start_index=True,
    )

    for file_path in pdf_files:
        try:
            docs = pdf_loader_cls(file_path).load()
            all_splits.extend(splitter.split_documents(docs))
        except Exception as exc:
            print(f"Error loading {file_path}: {exc}")
    return all_splits


def _build_chat_model(model_name: str, temperature: float):
    chat_ollama_cls = _get_chat_ollama_cls()
    kwargs = {
        "model": model_name,
        "base_url": SETTINGS.ollama_base_url,
        "temperature": temperature,
    }
    if SETTINGS.generation.max_tokens is not None:
        kwargs["num_predict"] = SETTINGS.generation.max_tokens
    if SETTINGS.generation.timeout_seconds is not None:
        kwargs["timeout"] = SETTINGS.generation.timeout_seconds
    return chat_ollama_cls(**kwargs)


def initialize_embeddings():
    embeddings_cls = _get_hf_embeddings_cls()
    return embeddings_cls(
        model_name=SETTINGS.embedding.model,
        model_kwargs={"device": SETTINGS.embedding.device},
        encode_kwargs={"normalize_embeddings": SETTINGS.embedding.normalize_embeddings},
    )


def build_vectorstore(all_splits: List[Document], embeddings, force_reindex: bool = False) -> Chroma:
    _require_dependency(chromadb, "chromadb")
    _require_dependency(Chroma, "langchain-chroma")
    client = chromadb.PersistentClient(path=str(SETTINGS.storage.chroma_persist_directory))
    collection = client.get_or_create_collection(name=SETTINGS.storage.collection_name)

    vector_store = Chroma(
        client=client,
        collection_name=SETTINGS.storage.collection_name,
        embedding_function=embeddings,
        persist_directory=str(SETTINGS.storage.chroma_persist_directory),
    )

    needs_reindex = force_reindex or collection.count() == 0

    if not needs_reindex and collection.count() > 0:
        try:
            vector_store.similarity_search("dimension check", k=1)
        except Exception as exc:
            msg = str(exc).lower()
            if "expecting embedding with dimension" in msg or "dimension" in msg:
                print(f"Detected embedding dimension mismatch. Reindex required: {exc}")
                needs_reindex = True
            else:
                raise

    if needs_reindex:
        if collection.count() > 0:
            client.delete_collection(name=SETTINGS.storage.collection_name)
            client.create_collection(name=SETTINGS.storage.collection_name)
            vector_store = Chroma(
                client=client,
                collection_name=SETTINGS.storage.collection_name,
                embedding_function=embeddings,
                persist_directory=str(SETTINGS.storage.chroma_persist_directory),
            )

        vector_store.add_documents(documents=all_splits)
        pdf_files = get_pdf_files(PDF_DIRECTORY)
        if pdf_files:
            save_pdf_hashes(get_pdf_hashes(pdf_files))

    return vector_store


def build_retriever(vector_store: Chroma, all_splits: List[Document]) -> Callable[[str, int], List[Document]]:
    bm25_cls = _get_bm25_retriever_cls()
    bm25 = bm25_cls.from_documents(all_splits)
    bm25.k = SETTINGS.retrieval.bm25_k
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": SETTINGS.retrieval.vector_k})

    def hybrid(query: str, k: Optional[int] = None) -> List[Document]:
        target_k = k or SETTINGS.retrieval.top_k
        bm25_docs = bm25.invoke(query)
        vs_docs = vector_retriever.invoke(query)

        combined: List[Document] = []
        seen = set()
        for doc in bm25_docs + vs_docs:
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])
            if key in seen:
                continue
            seen.add(key)
            combined.append(doc)
            if len(combined) >= target_k:
                break
        return combined

    return hybrid


def create_hybrid_retriever(all_splits: List[Document], embeddings, force_reindex: bool = False):
    vector_store = build_vectorstore(all_splits, embeddings, force_reindex=force_reindex)
    return build_retriever(vector_store, all_splits)


def initialize_reranker():
    return build_reranker()


def build_reranker():
    try:
        reranker_cls = _get_flag_reranker_cls()
    except Exception as exc:
        print(f"Warning: FlagEmbedding not available ({exc}). Trying sentence-transformers CrossEncoder fallback.")
        try:
            cross_encoder_cls = _get_cross_encoder_cls()
            model = cross_encoder_cls(SETTINGS.reranker.model)

            class _CrossEncoderAdapter:
                def __init__(self, ce_model):
                    self._ce_model = ce_model

                def compute_score(self, pairs):
                    return self._ce_model.predict(pairs).tolist()

            print(f"Info: Using CrossEncoder fallback reranker with model '{SETTINGS.reranker.model}'.")
            return _CrossEncoderAdapter(model)
        except Exception as ce_exc:
            print(
                "Warning: CrossEncoder fallback not available "
                f"({ce_exc}). Falling back to retrieval order."
            )
            return None
    try:
        return reranker_cls(SETTINGS.reranker.model, use_fp16=SETTINGS.reranker.use_fp16)
    except Exception as exc:
        print(f"Warning: Failed to initialize reranker '{SETTINGS.reranker.model}': {exc}")
        print("Falling back to retrieval order (no reranker).")
        return None


def rerank_documents(query: str, documents: List[Document], reranker_model, top_k: int = 3) -> List[Document]:
    if not documents:
        return []
    if reranker_model is None:
        return documents[:top_k]

    pairs = [[query, doc.page_content[: SETTINGS.reranker.max_doc_chars]] for doc in documents]
    scores = reranker_model.compute_score(pairs)
    if not isinstance(scores, list):
        scores = [scores]

    ranked_indices = sorted(
        range(len(documents)),
        key=lambda idx: float(scores[idx]),
        reverse=True,
    )[:top_k]

    return [documents[idx] for idx in ranked_indices]


def build_compressor():
    return _build_chat_model(SETTINGS.roles.compress, SETTINGS.generation.temperature_aux)


def initialize_compressor():
    return build_compressor()


def build_generation_chain(answer_llm):
    style = (ANSWER_STYLE or "detailed").strip().lower()
    if style == "detailed":
        task = "Produce a detailed, step-by-step answer. Then add a short source citation block."
    else:
        task = "Produce a concise answer (1-2 sentences). Then add a source citation block."

    prompt_template_cls = _get_chat_prompt_template_cls()
    prompt_template = prompt_template_cls.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT + "\nCONTEXT: {context}\nTASK: " + task,
            ),
            ("user", "{query}"),
        ]
    )
    return prompt_template | answer_llm


def initialize_llm():
    return _build_chat_model(SETTINGS.roles.answer, SETTINGS.generation.temperature_answer)


def initialize_router_llm():
    return _build_chat_model(SETTINGS.roles.router, SETTINGS.generation.temperature_aux)


def initialize_self_check_llm():
    return _build_chat_model(SETTINGS.roles.self_check, SETTINGS.generation.temperature_aux)


def initialize_query_rewrite_llm():
    return _build_chat_model(SETTINGS.roles.rewrite, SETTINGS.generation.temperature_aux)


def _format_chat_history(chat_history: Optional[List[dict]]) -> str:
    if not chat_history:
        return ""
    return "".join(
        f"User: {turn.get('query', '')}\nAssistant: {turn.get('answer', '')}\n"
        for turn in chat_history
    )


def decide_retrieval(router_llm, query: str, chat_history: Optional[List[dict]] = None) -> str:
    prompt_template_cls = _get_chat_prompt_template_cls()
    prompt_template = prompt_template_cls.from_messages(
        [
            ("system", ROUTER_SYSTEM_PROMPT),
            (
                "user",
                f"User Query: {query}\nChat History: {_format_chat_history(chat_history)}",
            ),
        ]
    )
    response = (prompt_template | router_llm).invoke({})
    return response.content.strip().upper()


def rewrite_query(query_rewrite_llm, query: str, chat_history: Optional[List[dict]] = None) -> str:
    system_msg = QUERY_REWRITE_SYSTEM_PROMPT + f"\nChat History:\n{_format_chat_history(chat_history)}"
    prompt_template_cls = _get_chat_prompt_template_cls()
    prompt = prompt_template_cls.from_messages(
        [
            ("system", system_msg),
            (
                "user",
                f"Original Query: {query}\nContext: (low relevance)\nTASK: Rewrite the query to improve retrieval results.",
            ),
        ]
    )
    return (prompt | query_rewrite_llm).invoke({}).content.strip()


def perform_self_check(
    self_check_llm,
    query_rewrite_llm,
    original_query: str,
    retrieved_docs: List[Document],
    chat_history: Optional[List[dict]] = None,
    max_retries: int = 2,
) -> Tuple[str, List[Document]]:
    current_query = original_query
    current_docs = retrieved_docs

    for _ in range(max_retries):
        if not current_docs:
            return rewrite_query(query_rewrite_llm, current_query, chat_history), []

        context_for_self_check = "\n---\n".join(doc.page_content for doc in current_docs)
        prompt_template_cls = _get_chat_prompt_template_cls()
        prompt = prompt_template_cls.from_messages(
            [
                ("system", SELF_CHECK_SYSTEM_PROMPT),
                ("user", f"User Query: {current_query}\nContext:\n{context_for_self_check}"),
            ]
        )

        decision = (prompt | self_check_llm).invoke({}).content.strip().upper()
        if decision == "RELEVANT":
            return current_query, current_docs

        current_query = rewrite_query(query_rewrite_llm, current_query, chat_history)
        return current_query, []

    return current_query, current_docs


def compress_context(compressor_llm, documents: List[Document], instruction: str) -> List[Document]:
    if not documents:
        return []

    merged_text = "\n\n---\n\n".join(doc.page_content for doc in documents)
    prompt_template_cls = _get_chat_prompt_template_cls()
    prompt = prompt_template_cls.from_messages(
        [
            (
                "system",
                f"Summarize the context so it is maximally relevant to the question. "
                "Limit to ~300 tokens and keep important numbers, exceptions, and definitions. "
                f"Write the summary in {RESPONSE_LANGUAGE}.",
            ),
            ("user", f"Question:\n{instruction}\n\nContext:\n{merged_text}"),
        ]
    )
    response = (prompt | compressor_llm).invoke({})
    return [Document(page_content=response.content, metadata={"source": "LLM-Compressed"})]


def generate_answer(
    llm,
    query: str,
    context_docs: List[Document],
    chat_history: Optional[List[dict]] = None,
) -> str:
    def _extract_text(response: Any) -> str:
        if response is None:
            return ""
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(p.strip() for p in parts if p and p.strip()).strip()
        return str(content).strip()

    def _compact_context(docs: List[Document], max_docs: int = 3, max_chars: int = 1200) -> str:
        snippets: List[str] = []
        for doc in docs[:max_docs]:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            page_suffix = f":{page}" if page is not None else ""
            body = (doc.page_content or "")[:max_chars].strip()
            snippets.append(f"[{source}{page_suffix}] {body}")
        return "\n\n".join(snippets)

    context = "\n---\n".join(doc.page_content for doc in context_docs)
    chain = build_generation_chain(llm)
    response = chain.invoke(
        {
            "query": query,
            "context": context,
            "chat_history": _format_chat_history(chat_history),
        }
    )
    answer = _extract_text(response)
    if answer:
        return answer

    # Some model/provider combinations occasionally return an empty payload.
    # Retry once with a simpler prompt and compact context to improve robustness.
    print("Warning: Empty LLM answer received. Retrying with fallback prompt.")
    prompt_template_cls = _get_chat_prompt_template_cls()
    fallback_prompt = prompt_template_cls.from_messages(
        [
            (
                "system",
                f"You are a helpful assistant. Answer using only the provided context in {RESPONSE_LANGUAGE}. "
                "If the context is insufficient, say so clearly.",
            ),
            ("user", "Question:\n{query}\n\nContext:\n{context}\n\nAnswer in 2-5 sentences."),
        ]
    )
    fallback_response = (fallback_prompt | llm).invoke(
        {
            "query": query,
            "context": _compact_context(context_docs),
        }
    )
    fallback_answer = _extract_text(fallback_response)
    if fallback_answer:
        return fallback_answer

    return (
        "I could not generate a reliable answer from the current model response. "
        "Please try again or rephrase the question."
    )


def document_to_source(doc: Document) -> Source:
    source_path = doc.metadata.get("source", "unknown")
    document_title = os.path.basename(source_path) if source_path != "unknown" else "Unknown document"
    document_id = Path(source_path).stem if source_path != "unknown" else "unknown"

    page = doc.metadata.get("page")
    if isinstance(page, str):
        try:
            page = int(page)
        except ValueError:
            page = None

    snippet = doc.page_content[:300] if doc.page_content else None
    return Source(
        document_id=document_id,
        document_title=document_title,
        page=page,
        section=None,
        snippet=snippet,
    )


class RAGPipeline:
    def __init__(self, settings: ModelSettings):
        self.settings = settings
        self.components: Optional[Dict[str, Any]] = None
        self._init_lock = threading.RLock()

    def build_vectorstore(self, all_splits: List[Document], embeddings, force_reindex: bool = False) -> Chroma:
        return build_vectorstore(all_splits, embeddings, force_reindex=force_reindex)

    def build_retriever(self, vector_store: Chroma, all_splits: List[Document]):
        return build_retriever(vector_store, all_splits)

    def build_reranker(self):
        return build_reranker()

    def build_compressor(self):
        return build_compressor()

    def build_generation_chain(self, llm):
        return build_generation_chain(llm)

    def initialize(self, force_reindex: bool = False) -> Dict[str, Any]:
        with self._init_lock:
            if self.components is not None and not force_reindex:
                return self.components

            os.makedirs(PDF_DIRECTORY, exist_ok=True)
            pdf_files = get_pdf_files(PDF_DIRECTORY)
            use_insuranceqa = os.getenv("USE_INSURANCEQA_DATA", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
            insuranceqa_mode = os.getenv("INSURANCEQA_RETRIEVAL_MODE", "merge").strip().lower()

            # PDF layer is optional when we explicitly run in InsuranceQA-only modes.
            vector_store: Optional[Chroma] = None
            hybrid_retriever: Callable[[str, Optional[int]], List[Document]]

            if pdf_files:
                all_splits = load_and_split_documents(pdf_files)
                embeddings = initialize_embeddings()

                reindex_required = force_reindex or embedding_model_has_changed()
                vector_store = self.build_vectorstore(all_splits, embeddings, force_reindex=reindex_required)
                hybrid_retriever = self.build_retriever(vector_store, all_splits)
            else:
                if not (use_insuranceqa and insuranceqa_mode in {"switch", "auto"}):
                    raise ValueError(f"No PDF files found in {PDF_DIRECTORY}. Please add insurance documents.")

                def _no_pdf_retriever(_query: str, k: Optional[int] = None) -> List[Document]:
                    del k
                    return []

                hybrid_retriever = _no_pdf_retriever

            insuranceqa_retriever = None
            if use_insuranceqa:
                try:
                    from src.data.insuranceqa_ingestion import (
                        INSURANCEQA_COLLECTION_NAME,
                        build_insuranceqa_index,
                        get_insuranceqa_retriever,
                    )

                    auto_build_insuranceqa = os.getenv("INSURANCEQA_AUTO_BUILD", "true").strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "y",
                        "on",
                    }
                    if auto_build_insuranceqa and _insuranceqa_collection_is_empty(INSURANCEQA_COLLECTION_NAME):
                        print("Info: InsuranceQA collection is empty. Building index from dataset...")
                        build_insuranceqa_index(force_reindex=False)

                    insuranceqa_retriever = get_insuranceqa_retriever()
                except Exception as exc:
                    print(f"Warning: Failed to initialize InsuranceQA retriever: {exc}")
                    insuranceqa_retriever = None

            if use_insuranceqa and insuranceqa_mode == "switch" and insuranceqa_retriever is None:
                raise RuntimeError(
                    "INSURANCEQA_RETRIEVAL_MODE=switch is active, but InsuranceQA retriever is unavailable. "
                    "Check dataset index build and dependencies."
                )

            if insuranceqa_retriever is not None:
                def _invoke_with_k(retriever, query: str, k: int) -> List[Document]:
                    # Many retrievers store 'k' in search_kwargs; update temporarily for this call.
                    search_kwargs = getattr(retriever, "search_kwargs", None)
                    if isinstance(search_kwargs, dict):
                        prev_k = search_kwargs.get("k")
                        search_kwargs["k"] = k
                        try:
                            return retriever.invoke(query)
                        finally:
                            if prev_k is None:
                                search_kwargs.pop("k", None)
                            else:
                                search_kwargs["k"] = prev_k
                    return retriever.invoke(query)

                base_retriever = hybrid_retriever

                def merged_retriever(query: str, k: Optional[int] = None) -> List[Document]:
                    target_k = k or SETTINGS.retrieval.top_k
                    base_docs = base_retriever(query, k=target_k)
                    qa_docs = _invoke_with_k(insuranceqa_retriever, query, target_k)

                    combined: List[Document] = []
                    seen = set()
                    for doc in base_docs + qa_docs:
                        key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])
                        if key in seen:
                            continue
                        seen.add(key)
                        combined.append(doc)
                        if len(combined) >= target_k:
                            break
                    return combined

                def insuranceqa_only(query: str, k: Optional[int] = None) -> List[Document]:
                    target_k = k or SETTINGS.retrieval.top_k
                    return _invoke_with_k(insuranceqa_retriever, query, target_k)

                def insuranceqa_auto(query: str, k: Optional[int] = None) -> List[Document]:
                    # Route InsuranceQA-style questions to the InsuranceQA retriever,
                    # otherwise use the PDF retriever (or merged behavior if desired).
                    if _should_route_to_insuranceqa(query):
                        return insuranceqa_only(query, k=k)
                    return base_retriever(query, k=k)

                if insuranceqa_mode == "switch":
                    hybrid_retriever = insuranceqa_only
                elif insuranceqa_mode == "auto":
                    hybrid_retriever = insuranceqa_auto
                else:
                    hybrid_retriever = merged_retriever

            self.components = {
                "vector_store": vector_store,
                "hybrid_retriever": hybrid_retriever,
                "insuranceqa_retriever": insuranceqa_retriever,
                "reranker_model": self.build_reranker(),
                "compressor_llm": self.build_compressor(),
                "llm": initialize_llm(),
                "router_llm": initialize_router_llm(),
                "self_check_llm": initialize_self_check_llm(),
                "query_rewrite_llm": initialize_query_rewrite_llm(),
                "generation_chain": self.build_generation_chain(initialize_llm()),
                "safety_checker": SafetyAuditLayer(self.settings.safety),
            }

            _save_model_config()
            return self.components

    def run(self, query: str, chat_history: Optional[List[dict]] = None) -> AnswerResult:
        start_time = datetime.now()
        chat_history = chat_history or []
        original_query = query
        local_safety_checker = SafetyAuditLayer(self.settings.safety)
        safety_pre_result = _default_allow_safety_result("pre_query")
        safety_context_result = _default_allow_safety_result("context")
        safety_post_result = _default_allow_safety_result("post_generation")
        final_safety_decision = "allow"
        safety_system_error = False
        safety_error_stage: Optional[str] = None
        safety_error_type: Optional[str] = None
        safety_error_message: Optional[str] = None

        def _mark_safety_system_error(stage: str, exc: Exception) -> None:
            nonlocal safety_system_error, safety_error_stage, safety_error_type, safety_error_message
            safety_system_error = True
            safety_error_stage = stage
            safety_error_type = type(exc).__name__
            safety_error_message = str(exc)

        try:
            safety_pre_result = local_safety_checker.check_query_safety(query, chat_history)
        except Exception as exc:
            _mark_safety_system_error("pre_query", exc)
            if self.settings.safety.fail_closed and local_safety_checker.is_active:
                safety_pre_result = SafetyResult(
                    allow=False,
                    risk_level="high",
                    reasons=["safety_precheck_error"],
                    action="fallback",
                    details={"stage": "pre_query", "error": str(exc)},
                )

        if not safety_pre_result.allow:
            blocked_answer = local_safety_checker.apply_safety_action(safety_pre_result, "")
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            if safety_system_error and safety_error_stage == "pre_query":
                final_safety_decision = f"pre_system_error_{safety_pre_result.action}"
            else:
                final_safety_decision = f"pre_{safety_pre_result.action}"
            audit_log(
                query=original_query,
                retrieved_documents=[],
                compressed_context=[],
                generated_answer=blocked_answer,
                chat_history=chat_history,
                retrieval_needed="SAFETY_PRECHECK_BLOCKED",
                final_query=original_query,
                sources=[],
                latency_ms=latency_ms,
                retries=0,
                provider=self.settings.provider,
                answer_style=ANSWER_STYLE,
                response_language=RESPONSE_LANGUAGE,
                safety_mode=self.settings.safety.mode,
                safety_enabled=local_safety_checker.is_active,
                safety_decision=final_safety_decision,
                safety_risks=list(safety_pre_result.reasons),
                safety_scores={"pre": dict(safety_pre_result.scores)},
                safety_block_reason=safety_pre_result.reasons[0] if safety_pre_result.reasons else "precheck_blocked",
                safety_results={
                    "pre": _safety_result_to_dict(safety_pre_result),
                    "context": _safety_result_to_dict(safety_context_result),
                    "post": _safety_result_to_dict(safety_post_result),
                },
                safety_system_error=safety_system_error,
                safety_error_stage=safety_error_stage,
                safety_error_type=safety_error_type,
                safety_error_message=safety_error_message,
            )
            return AnswerResult(answer=blocked_answer, sources=[], query=query, latency_ms=latency_ms)

        if _insuranceqa_exact_match_enabled():
            exact_answer = lookup_insuranceqa_answer(query)
            if exact_answer:
                exact_doc = Document(
                    page_content=f"Question: {query}\nAnswer: {exact_answer}",
                    metadata={"source": "insuranceqa_v2_local", "dataset": "InsuranceQA", "exact_match": True},
                )
                sources = [
                    Source(
                        document_id="insuranceqa_v2_local",
                        document_title="insuranceQA-v2 (local JSONL)",
                        page=None,
                        section="exact-match",
                        snippet=exact_answer[:300],
                    )
                ]
                final_answer = exact_answer
                try:
                    safety_post_result = local_safety_checker.check_answer_safety(query, [exact_doc], exact_answer)
                except Exception as exc:
                    _mark_safety_system_error("post_generation", exc)
                    if self.settings.safety.fail_closed and local_safety_checker.is_active:
                        safety_post_result = SafetyResult(
                            allow=False,
                            risk_level="high",
                            reasons=["safety_postcheck_error"],
                            action="fallback",
                            details={"stage": "post_generation", "error": str(exc)},
                        )

                if not safety_post_result.allow:
                    final_answer = local_safety_checker.apply_safety_action(safety_post_result, exact_answer)
                    if safety_post_result.action in {"block", "fallback"}:
                        sources = []
                    if safety_system_error and safety_error_stage == "post_generation":
                        final_safety_decision = f"post_system_error_{safety_post_result.action}"
                    else:
                        final_safety_decision = f"post_{safety_post_result.action}"

                latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                audit_log(
                    query=original_query,
                    retrieved_documents=[exact_doc],
                    compressed_context=[exact_doc],
                    generated_answer=final_answer,
                    chat_history=chat_history,
                    retrieval_needed="RETRIEVE",
                    final_query=original_query,
                    sources=[s.__dict__ for s in sources],
                    latency_ms=latency_ms,
                    retries=0,
                    provider=self.settings.provider,
                    answer_style=ANSWER_STYLE,
                    response_language=RESPONSE_LANGUAGE,
                    exact_insuranceqa_match=True,
                    safety_mode=self.settings.safety.mode,
                    safety_enabled=local_safety_checker.is_active,
                    safety_decision=final_safety_decision,
                    safety_risks=list(set(safety_pre_result.reasons + safety_post_result.reasons)),
                    safety_scores={
                        "pre": dict(safety_pre_result.scores),
                        "context": dict(safety_context_result.scores),
                        "post": dict(safety_post_result.scores),
                    },
                    safety_block_reason=safety_post_result.reasons[0]
                    if (not safety_post_result.allow and safety_post_result.reasons)
                    else None,
                    safety_results={
                        "pre": _safety_result_to_dict(safety_pre_result),
                        "context": _safety_result_to_dict(safety_context_result),
                        "post": _safety_result_to_dict(safety_post_result),
                    },
                    safety_system_error=safety_system_error,
                    safety_error_stage=safety_error_stage,
                    safety_error_type=safety_error_type,
                    safety_error_message=safety_error_message,
                )
                return AnswerResult(answer=final_answer, sources=sources, query=query, latency_ms=latency_ms)

        components = self.initialize(force_reindex=pdfs_have_changed())
        safety_checker = components.get("safety_checker", local_safety_checker)

        if self.settings.retrieval.force_retrieval:
            retrieval_needed = "RETRIEVE"
        else:
            retrieval_needed = decide_retrieval(components["router_llm"], query, chat_history)

        if retrieval_needed != "RETRIEVE":
            answer = "This question does not require document retrieval, or it can be answered without searching the document base."
            sources: List[Source] = []
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            audit_log(
                query=original_query,
                retrieved_documents=[],
                compressed_context=[],
                generated_answer=answer,
                chat_history=chat_history,
                retrieval_needed=retrieval_needed,
                final_query=original_query,
                sources=[],
                latency_ms=latency_ms,
                provider=self.settings.provider,
                answer_style=ANSWER_STYLE,
                response_language=RESPONSE_LANGUAGE,
                safety_mode=self.settings.safety.mode,
                safety_enabled=safety_checker.is_active,
                safety_decision=final_safety_decision,
                safety_risks=list(safety_pre_result.reasons),
                safety_scores={
                    "pre": dict(safety_pre_result.scores),
                    "context": dict(safety_context_result.scores),
                    "post": dict(safety_post_result.scores),
                },
                safety_block_reason=None,
                safety_results={
                    "pre": _safety_result_to_dict(safety_pre_result),
                    "context": _safety_result_to_dict(safety_context_result),
                    "post": _safety_result_to_dict(safety_post_result),
                },
                safety_system_error=safety_system_error,
                safety_error_stage=safety_error_stage,
                safety_error_type=safety_error_type,
                safety_error_message=safety_error_message,
            )
            return AnswerResult(answer=answer, sources=sources, query=query, latency_ms=latency_ms)

        current_query = query
        reranked_docs: List[Document] = []
        last_retrieved_docs: List[Document] = []

        retries = 0
        while retries < self.settings.retrieval.max_self_check_retries:
            retrieved_docs = components["hybrid_retriever"](current_query, k=self.settings.retrieval.top_k)
            last_retrieved_docs = retrieved_docs
            reranked_docs = rerank_documents(
                current_query,
                retrieved_docs,
                components["reranker_model"],
                top_k=self.settings.reranker.top_k,
            )

            if not reranked_docs:
                current_query = rewrite_query(components["query_rewrite_llm"], current_query, chat_history)
                retries += 1
                continue

            checked_query, checked_docs = perform_self_check(
                components["self_check_llm"],
                components["query_rewrite_llm"],
                current_query,
                reranked_docs,
                chat_history,
                max_retries=1,
            )

            if checked_query != current_query or not checked_docs:
                current_query = checked_query
                retries += 1
                continue

            reranked_docs = checked_docs
            break

        if not reranked_docs:
            answer = (
                "Sorry, I couldn't find relevant information in the indexed documents. "
                "Please rephrase your question or provide additional documents."
            )
            sources = []
            context_docs_for_log: List[Document] = []
        else:
            context_docs = reranked_docs
            try:
                safety_context_result = safety_checker.check_context_safety(reranked_docs)
            except Exception as exc:
                _mark_safety_system_error("context", exc)
                if self.settings.safety.fail_closed and safety_checker.is_active:
                    safety_context_result = SafetyResult(
                        allow=False,
                        risk_level="high",
                        reasons=["safety_context_check_error"],
                        action="fallback",
                        details={"stage": "context", "error": str(exc)},
                    )

            if not safety_context_result.allow:
                answer = safety_checker.apply_safety_action(safety_context_result, "")
                sources = []
                context_docs_for_log = reranked_docs
                if safety_system_error and safety_error_stage == "context":
                    final_safety_decision = f"context_system_error_{safety_context_result.action}"
                else:
                    final_safety_decision = f"context_{safety_context_result.action}"
            else:
                if self.settings.retrieval.enable_context_compression:
                    context_docs = compress_context(components["compressor_llm"], reranked_docs, current_query)

                answer = generate_answer(components["llm"], current_query, context_docs, chat_history)
                sources = [document_to_source(doc) for doc in reranked_docs]
                context_docs_for_log = context_docs

                try:
                    safety_post_result = safety_checker.check_answer_safety(current_query, context_docs, answer)
                except Exception as exc:
                    _mark_safety_system_error("post_generation", exc)
                    if self.settings.safety.fail_closed and safety_checker.is_active:
                        safety_post_result = SafetyResult(
                            allow=False,
                            risk_level="high",
                            reasons=["safety_postcheck_error"],
                            action="fallback",
                            details={"stage": "post_generation", "error": str(exc)},
                        )

                if not safety_post_result.allow:
                    answer = safety_checker.apply_safety_action(safety_post_result, answer)
                    if safety_post_result.action in {"block", "fallback"}:
                        sources = []
                    if safety_system_error and safety_error_stage == "post_generation":
                        final_safety_decision = f"post_system_error_{safety_post_result.action}"
                    else:
                        final_safety_decision = f"post_{safety_post_result.action}"

        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        safety_risks = list(
            set(
                safety_pre_result.reasons
                + safety_context_result.reasons
                + safety_post_result.reasons
            )
        )
        audit_log(
            query=original_query,
            retrieved_documents=last_retrieved_docs,
            compressed_context=context_docs_for_log,
            generated_answer=answer,
            chat_history=chat_history,
            retrieval_needed=retrieval_needed,
            final_query=current_query,
            sources=[s.__dict__ for s in sources],
            latency_ms=latency_ms,
            retries=retries,
            provider=self.settings.provider,
            answer_style=ANSWER_STYLE,
            response_language=RESPONSE_LANGUAGE,
            safety_mode=self.settings.safety.mode,
            safety_enabled=safety_checker.is_active,
            safety_decision=final_safety_decision,
            safety_risks=safety_risks,
            safety_scores={
                "pre": dict(safety_pre_result.scores),
                "context": dict(safety_context_result.scores),
                "post": dict(safety_post_result.scores),
            },
            safety_block_reason=safety_risks[0] if safety_risks and final_safety_decision != "allow" else None,
            safety_results={
                "pre": _safety_result_to_dict(safety_pre_result),
                "context": _safety_result_to_dict(safety_context_result),
                "post": _safety_result_to_dict(safety_post_result),
            },
            safety_system_error=safety_system_error,
            safety_error_stage=safety_error_stage,
            safety_error_type=safety_error_type,
            safety_error_message=safety_error_message,
        )
        return AnswerResult(answer=answer, sources=sources, query=query, latency_ms=latency_ms)


def initialize_pipeline(force_reindex: bool = False):
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = RAGPipeline(SETTINGS)
        return _pipeline.initialize(force_reindex=force_reindex)


def run_rag(question: str, chat_history: Optional[List[dict]] = None) -> AnswerResult:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = RAGPipeline(SETTINGS)
        pipeline = _pipeline
    return pipeline.run(question, chat_history=chat_history)


def is_pipeline_ready() -> bool:
    with _pipeline_lock:
        return _pipeline is not None and _pipeline.components is not None
