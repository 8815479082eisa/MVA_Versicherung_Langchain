from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

Provider = Literal["ollama"]
SafetyMode = Literal["off", "monitor", "enforce"]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class RoleModels:
    answer: str
    router: str
    self_check: str
    rewrite: str
    compress: str


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    device: str
    normalize_embeddings: bool


@dataclass(frozen=True)
class RerankerConfig:
    model: str
    use_fp16: bool
    top_k: int
    max_doc_chars: int


@dataclass(frozen=True)
class GenerationConfig:
    temperature_answer: float
    temperature_aux: float
    max_tokens: Optional[int]
    timeout_seconds: Optional[float]


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    bm25_k: int
    vector_k: int
    max_self_check_retries: int
    force_retrieval: bool
    enable_context_compression: bool


@dataclass(frozen=True)
class StorageConfig:
    project_root: Path
    data_root: Path
    raw_data_root: Path
    processed_data_root: Path
    benchmark_root: Path
    pdf_directory: Path
    chroma_persist_directory: Path
    collection_name: str
    pdf_hash_file: Path
    model_config_file: Path
    audit_log_file: Path


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class SafetyConfig:
    enabled: bool
    mode: SafetyMode
    min_groundedness: float
    block_pii: bool
    block_injection: bool
    fail_closed: bool
    fallback_text: str


@dataclass(frozen=True)
class ModelSettings:
    provider: Provider
    ollama_base_url: str
    roles: RoleModels
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    generation: GenerationConfig
    retrieval: RetrievalConfig
    storage: StorageConfig
    chunking: ChunkingConfig
    safety: SafetyConfig


def load_model_settings() -> ModelSettings:
    project_root = Path(__file__).resolve().parents[2]
    data_root = Path(os.getenv("DATA_DIR", str(project_root / "data")))
    raw_data_root = Path(os.getenv("RAW_DATA_DIR", str(data_root / "raw")))
    processed_data_root = Path(
        os.getenv("PROCESSED_DATA_DIR", str(data_root / "processed"))
    )
    benchmark_root = Path(
        os.getenv("BENCHMARK_DIR", str(data_root / "benchmarks"))
    )

    # Runtime is local-only: always use Ollama models.
    provider: Provider = "ollama"

    default_ollama_model = os.getenv("OLLAMA_MODEL", "rnj-1:8b")
    roles = RoleModels(
        answer=os.getenv("ANSWER_MODEL", default_ollama_model),
        router=os.getenv("ROUTER_MODEL", default_ollama_model),
        self_check=os.getenv("SELF_CHECK_MODEL", default_ollama_model),
        rewrite=os.getenv("QUERY_REWRITE_MODEL", default_ollama_model),
        compress=os.getenv("COMPRESSOR_MODEL", default_ollama_model),
    )

    embedding = EmbeddingConfig(
        model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        normalize_embeddings=_env_bool("EMBEDDING_NORMALIZE", True),
    )

    reranker = RerankerConfig(
        model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        use_fp16=_env_bool("RERANKER_USE_FP16", False),
        top_k=_env_int("RERANK_TOP_K", 5),
        max_doc_chars=_env_int("RERANK_MAX_DOC_CHARS", 2500),
    )

    max_tokens = _env_int("LLM_MAX_TOKENS", 0)
    timeout_seconds = _env_float("LLM_TIMEOUT_SECONDS", 0.0)

    generation = GenerationConfig(
        temperature_answer=_env_float("ANSWER_TEMPERATURE", 0.1),
        temperature_aux=_env_float("AUX_TEMPERATURE", 0.0),
        max_tokens=max_tokens if max_tokens > 0 else None,
        timeout_seconds=timeout_seconds if timeout_seconds > 0 else None,
    )

    retrieval = RetrievalConfig(
        top_k=_env_int("RETRIEVE_TOP_K", 8),
        bm25_k=_env_int("BM25_TOP_K", 5),
        vector_k=_env_int("VECTOR_TOP_K", 5),
        max_self_check_retries=_env_int("MAX_SELF_CHECK_RETRIES", 2),
        force_retrieval=_env_bool("RAG_FORCE_RETRIEVAL", True),
        enable_context_compression=_env_bool("ENABLE_CONTEXT_COMPRESSION", False),
    )

    storage = StorageConfig(
        project_root=project_root,
        data_root=data_root,
        raw_data_root=raw_data_root,
        processed_data_root=processed_data_root,
        benchmark_root=benchmark_root,
        pdf_directory=Path(
            os.getenv("PDF_DIRECTORY", str(raw_data_root / "pdfs"))
        ),
        chroma_persist_directory=Path(
            os.getenv(
                "CHROMA_PERSIST_DIRECTORY",
                str(processed_data_root / "vectorstores" / "chroma_db"),
            )
        ),
        collection_name=os.getenv("COLLECTION_NAME", "insurance_rag_collection"),
        pdf_hash_file=Path(
            os.getenv(
                "PDF_HASH_FILE",
                str(processed_data_root / "caches" / "pdf_hashes.json"),
            )
        ),
        model_config_file=Path(
            os.getenv(
                "MODEL_CONFIG_FILE",
                str(processed_data_root / "caches" / "model_config.json"),
            )
        ),
        audit_log_file=Path(
            os.getenv(
                "AUDIT_LOG_FILE",
                str(processed_data_root / "logs" / "audit.log"),
            )
        ),
    )

    chunking = ChunkingConfig(
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
    )

    raw_mode = os.getenv("SAFETY_MODE", "monitor").strip().lower()
    safety_mode: SafetyMode
    if raw_mode in {"off", "monitor", "enforce"}:
        safety_mode = raw_mode  # type: ignore[assignment]
    else:
        safety_mode = "monitor"

    min_groundedness = _env_float("SAFETY_MIN_GROUNDEDNESS", 0.7)
    if min_groundedness < 0.0:
        min_groundedness = 0.0
    if min_groundedness > 1.0:
        min_groundedness = 1.0

    safety = SafetyConfig(
        enabled=_env_bool("SAFETY_ENABLED", True),
        mode=safety_mode,
        min_groundedness=min_groundedness,
        block_pii=_env_bool("SAFETY_BLOCK_PII", True),
        block_injection=_env_bool("SAFETY_BLOCK_INJECTION", True),
        fail_closed=_env_bool("SAFETY_FAIL_CLOSED", True),
        fallback_text=os.getenv(
            "SAFETY_FALLBACK_TEXT",
            "I cannot provide a safe, policy-compliant answer for this request. Please rephrase.",
        ).strip(),
    )

    return ModelSettings(
        provider=provider,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        roles=roles,
        embedding=embedding,
        reranker=reranker,
        generation=generation,
        retrieval=retrieval,
        storage=storage,
        chunking=chunking,
        safety=safety,
    )
