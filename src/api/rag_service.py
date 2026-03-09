"""
Unified RAG service used by API and CLI.

Migration note:
- `src/api/rag_service.py` is the single source of truth for the active pipeline.
- Legacy OpenAI-heavy orchestration was retired from the active runtime path.
- Compatibility wrappers remain available in `main.py` for evaluation scripts.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_chroma import Chroma
except Exception:
    Chroma = None

try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    PyPDFLoader = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None

try:
    from langchain_community.retrievers import BM25Retriever
except Exception:
    BM25Retriever = None

try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None

try:
    from langchain_openai import ChatOpenAI
except Exception:  # optional provider
    ChatOpenAI = None

try:
    from FlagEmbedding import FlagReranker
except Exception:
    FlagReranker = None

try:
    import chromadb
except Exception:
    chromadb = None

try:
    from config.models import ModelSettings, load_model_settings
except Exception:
    from src.config.models import ModelSettings, load_model_settings


# Prefer the project .env over inherited shell variables so the active runtime
# stays aligned with the repo configuration.
load_dotenv(override=True)
SETTINGS: ModelSettings = load_model_settings()

PDF_DIRECTORY = str(SETTINGS.storage.pdf_directory)
AUDIT_LOG_FILE = str(SETTINGS.storage.audit_log_file)
RESPONSE_LANGUAGE = os.getenv("RESPONSE_LANGUAGE", "English")
ANSWER_STYLE = os.getenv("ANSWER_STYLE", "detailed")  # detailed | concise


@dataclass
class Source:
    document_id: str
    document_title: str
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: Optional[str] = None


@dataclass
class AnswerResult:
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


_pipeline: Optional["RAGPipeline"] = None

_insuranceqa_questions_loaded = False
_insuranceqa_norm_questions: set[str] = set()
_insuranceqa_norm_questions_list: list[str] = []


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
    global _insuranceqa_questions_loaded, _insuranceqa_norm_questions, _insuranceqa_norm_questions_list
    if _insuranceqa_questions_loaded:
        return

    path_raw = os.getenv("INSURANCEQA_ROUTING_JSONL", "").strip()
    path = Path(path_raw) if path_raw else _default_insuranceqa_jsonl_path()
    if not path.exists():
        _insuranceqa_questions_loaded = True
        return

    norm_questions: list[str] = []
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
    except Exception:
        # Routing is best-effort. If it fails, fall back to default behavior.
        _insuranceqa_questions_loaded = True
        return

    _insuranceqa_norm_questions = set(norm_questions)
    _insuranceqa_norm_questions_list = norm_questions
    _insuranceqa_questions_loaded = True


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


def _require_dependency(dep, package_name: str) -> None:
    if dep is None:
        raise RuntimeError(
            f"Missing optional dependency '{package_name}'. "
            f"Install project requirements to use this feature."
        )


def _normalize_pdf_key(file_path: str) -> str:
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


def _load_model_config() -> dict:
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
    _require_dependency(PyPDFLoader, "langchain-community")
    _require_dependency(RecursiveCharacterTextSplitter, "langchain-text-splitters")
    all_splits: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunking.chunk_size,
        chunk_overlap=SETTINGS.chunking.chunk_overlap,
        add_start_index=True,
    )

    for file_path in pdf_files:
        try:
            docs = PyPDFLoader(file_path).load()
            all_splits.extend(splitter.split_documents(docs))
        except Exception as exc:
            print(f"Error loading {file_path}: {exc}")
    return all_splits


def _build_chat_model(model_name: str, temperature: float):
    if SETTINGS.provider == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("MODEL_PROVIDER=openai is set but langchain_openai is not installed.")
        kwargs = {"model": model_name, "temperature": temperature}
        if SETTINGS.generation.max_tokens is not None:
            kwargs["max_tokens"] = SETTINGS.generation.max_tokens
        if SETTINGS.generation.timeout_seconds is not None:
            kwargs["timeout"] = SETTINGS.generation.timeout_seconds
        return ChatOpenAI(**kwargs)

    kwargs = {
        "model": model_name,
        "base_url": SETTINGS.ollama_base_url,
        "temperature": temperature,
    }
    _require_dependency(ChatOllama, "langchain-ollama")
    if SETTINGS.generation.max_tokens is not None:
        kwargs["num_predict"] = SETTINGS.generation.max_tokens
    if SETTINGS.generation.timeout_seconds is not None:
        kwargs["timeout"] = SETTINGS.generation.timeout_seconds
    return ChatOllama(**kwargs)


def initialize_embeddings():
    _require_dependency(HuggingFaceEmbeddings, "langchain-community")
    return HuggingFaceEmbeddings(
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
    _require_dependency(BM25Retriever, "langchain-community")
    bm25 = BM25Retriever.from_documents(all_splits)
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
    if FlagReranker is None:
        print("Warning: FlagEmbedding not available. Falling back to retrieval order.")
        return None
    try:
        return FlagReranker(SETTINGS.reranker.model, use_fp16=SETTINGS.reranker.use_fp16)
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

    prompt_template = ChatPromptTemplate.from_messages(
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
    prompt_template = ChatPromptTemplate.from_messages(
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
    prompt = ChatPromptTemplate.from_messages(
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
        prompt = ChatPromptTemplate.from_messages(
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
    prompt = ChatPromptTemplate.from_messages(
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
    context = "\n---\n".join(doc.page_content for doc in context_docs)
    chain = build_generation_chain(llm)
    response = chain.invoke(
        {
            "query": query,
            "context": context,
            "chat_history": _format_chat_history(chat_history),
        }
    )
    return response.content


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
        }

        _save_model_config()
        return self.components

    def run(self, query: str, chat_history: Optional[List[dict]] = None) -> AnswerResult:
        start_time = datetime.now()
        chat_history = chat_history or []
        original_query = query

        components = self.initialize(force_reindex=pdfs_have_changed())

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
            if self.settings.retrieval.enable_context_compression:
                context_docs = compress_context(components["compressor_llm"], reranked_docs, current_query)

            answer = generate_answer(components["llm"], current_query, context_docs, chat_history)
            sources = [document_to_source(doc) for doc in reranked_docs]
            context_docs_for_log = context_docs

        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
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
        )
        return AnswerResult(answer=answer, sources=sources, query=query, latency_ms=latency_ms)


def initialize_pipeline(force_reindex: bool = False):
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(SETTINGS)
    return _pipeline.initialize(force_reindex=force_reindex)


def run_rag(question: str, chat_history: Optional[List[dict]] = None) -> AnswerResult:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(SETTINGS)
    return _pipeline.run(question, chat_history=chat_history)
