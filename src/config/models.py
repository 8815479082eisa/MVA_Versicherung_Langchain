from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - optional in some tooling contexts
    dotenv_values = None

Provider = Literal["ollama"]
SafetyMode = Literal["off", "monitor", "enforce"]
SafetyBackend = Literal["nemo", "off"]

_DEFAULT_PREFERRED_ANSWER_MODEL = "qwen3.5:4b"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DOTENV_PATH = _PROJECT_ROOT / ".env"
_ENV_SOURCES: Dict[str, str] = {}
_DOTENV_VALUES: Dict[str, str] = {}
_DOTENV_CONFLICTS: Dict[str, Dict[str, str]] = {}


def bootstrap_environment() -> None:
    global _DOTENV_VALUES, _DOTENV_CONFLICTS

    _DOTENV_CONFLICTS = {}
    if dotenv_values is None or not _DOTENV_PATH.exists():
        _DOTENV_VALUES = {}
        return

    parsed = {
        str(key): str(value)
        for key, value in dotenv_values(_DOTENV_PATH).items()
        if key and value is not None
    }
    _DOTENV_VALUES = parsed

    for key, value in parsed.items():
        current = os.environ.get(key)
        previous_source = _ENV_SOURCES.get(key)

        if current is None:
            os.environ[key] = value
            _ENV_SOURCES[key] = "dotenv"
            continue

        if previous_source == "dotenv" and current == value:
            _ENV_SOURCES[key] = "dotenv"
            continue

        _ENV_SOURCES[key] = "shell_env"
        if current != value:
            _DOTENV_CONFLICTS[key] = {"shell_env": current, "dotenv": value}


def _get_env_value(name: str) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def _env_source(name: str) -> str:
    if name in os.environ:
        return _ENV_SOURCES.get(name, "shell_env")
    return "default"


def _resolve_model_source(
    env_name: str,
    *,
    fallback_env_name: Optional[str] = None,
    default_value: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    raw_value = _get_env_value(env_name)
    if raw_value is not None:
        return _env_source(env_name), None

    if fallback_env_name is not None:
        fallback_value = _get_env_value(fallback_env_name)
        if fallback_value is not None:
            return "fallback", f"{fallback_env_name} ({_env_source(fallback_env_name)})"

    if default_value is not None:
        return "default", f"code default ({default_value})"
    return "default", None


def runtime_config_snapshot(settings: "ModelSettings | None" = None) -> Dict[str, Any]:
    bootstrap_environment()

    current_settings = settings or load_model_settings()
    configured_answer_model_source, configured_answer_model_source_detail = _resolve_model_source(
        "ANSWER_MODEL",
        fallback_env_name="PREFERRED_ANSWER_MODEL",
        default_value=_DEFAULT_PREFERRED_ANSWER_MODEL,
    )
    preferred_answer_model_source, preferred_answer_model_source_detail = _resolve_model_source(
        "PREFERRED_ANSWER_MODEL",
        default_value=_DEFAULT_PREFERRED_ANSWER_MODEL,
    )
    router_model_source, router_model_source_detail = _resolve_model_source(
        "ROUTER_MODEL",
        fallback_env_name="OLLAMA_MODEL",
        default_value=current_settings.preferred_answer_model,
    )
    ollama_model_source, ollama_model_source_detail = _resolve_model_source(
        "OLLAMA_MODEL",
        fallback_env_name="PREFERRED_ANSWER_MODEL",
        default_value=_DEFAULT_PREFERRED_ANSWER_MODEL,
    )

    answer_model_matches_preference = (
        current_settings.roles.answer == current_settings.preferred_answer_model
    )

    raw_env = {
        "ANSWER_MODEL": os.getenv("ANSWER_MODEL"),
        "PREFERRED_ANSWER_MODEL": os.getenv("PREFERRED_ANSWER_MODEL"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL"),
        "ROUTER_MODEL": os.getenv("ROUTER_MODEL"),
        "SAFETY_BACKEND": os.getenv("SAFETY_BACKEND"),
        "NEMO_ENFORCE_OUTPUT": os.getenv("NEMO_ENFORCE_OUTPUT"),
        "QUERY_REWRITE_ENABLED": os.getenv("QUERY_REWRITE_ENABLED"),
        "INSURANCEQA_EXACT_MATCH_SHORTCUT": os.getenv("INSURANCEQA_EXACT_MATCH_SHORTCUT"),
    }

    return {
        "raw_env": raw_env,
        "dotenv_path": str(_DOTENV_PATH) if _DOTENV_PATH.exists() else None,
        "dotenv_conflicts": dict(_DOTENV_CONFLICTS),
        "configured_answer_model": current_settings.roles.answer,
        "configured_answer_model_source": configured_answer_model_source,
        "configured_answer_model_source_detail": configured_answer_model_source_detail,
        "source_of_answer_model": configured_answer_model_source,
        "preferred_answer_model": current_settings.preferred_answer_model,
        "preferred_answer_model_source": preferred_answer_model_source,
        "preferred_answer_model_source_detail": preferred_answer_model_source_detail,
        "ollama_model": raw_env["OLLAMA_MODEL"],
        "ollama_model_source": ollama_model_source,
        "ollama_model_source_detail": ollama_model_source_detail,
        "router_model": current_settings.roles.router,
        "router_model_source": router_model_source,
        "router_model_source_detail": router_model_source_detail,
        "query_rewrite_enabled": current_settings.retrieval.query_rewrite_enabled,
        "nemo_enforce_output": current_settings.safety.nemo_enforce_output,
        "safety_backend": current_settings.safety.backend,
        "answer_model_matches_preference": answer_model_matches_preference,
    }


def build_answer_model_warning(settings: "ModelSettings") -> Optional[str]:
    snapshot = runtime_config_snapshot(settings)
    if snapshot["answer_model_matches_preference"]:
        return None

    warning = (
        f"Configured answer model '{settings.roles.answer}' differs from preferred model "
        f"'{settings.preferred_answer_model}'. Source: "
        f"{snapshot['configured_answer_model_source']}"
    )
    detail = snapshot.get("configured_answer_model_source_detail")
    if detail:
        warning += f" via {detail}"

    answer_model_conflict = snapshot.get("dotenv_conflicts", {}).get("ANSWER_MODEL")
    if answer_model_conflict:
        warning += (
            f". Shell env='{answer_model_conflict['shell_env']}', "
            f".env='{answer_model_conflict['dotenv']}'"
        )

    return warning


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


def _env_csv(name: str) -> Tuple[str, ...]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def resolve_preferred_answer_model() -> str:
    raw = _get_env_value("PREFERRED_ANSWER_MODEL")
    return raw or _DEFAULT_PREFERRED_ANSWER_MODEL


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
    query_rewrite_enabled: bool
    query_rewrite_min_similarity: float


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
    security_fallback_text: str = "I cannot provide a safe, policy-compliant answer for this request. Please rephrase."
    grounding_fallback_text: str = "I could not generate an answer that is sufficiently supported by the available documents."
    pii_fallback_text: str = "I cannot include or expose personal or sensitive information in the response."
    context_fallback_text: str = "The retrieved context contained information that could not be safely used for answering."
    pii_allowed_emails: Tuple[str, ...] = field(default_factory=tuple)
    pii_allowed_phone_numbers: Tuple[str, ...] = field(default_factory=tuple)
    pii_allowed_domains: Tuple[str, ...] = field(default_factory=tuple)
    pii_allowed_phone_prefixes: Tuple[str, ...] = field(default_factory=tuple)

    backend: SafetyBackend = "nemo"
    nemo_config_path: Optional[Path] = None
    nemo_context_config_path: Optional[Path] = None
    nemo_output_config_path: Optional[Path] = None
    nemo_input_enabled: bool = True
    nemo_context_enabled: bool = True
    nemo_output_enabled: bool = True
    nemo_enforce_input: bool = True
    nemo_enforce_output: bool = True
    nemo_runtime_timeout_seconds: float = 20.0


@dataclass(frozen=True)
class ModelSettings:
    provider: Provider
    ollama_base_url: str
    preferred_answer_model: str
    roles: RoleModels
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    generation: GenerationConfig
    retrieval: RetrievalConfig
    storage: StorageConfig
    chunking: ChunkingConfig
    safety: SafetyConfig


def load_model_settings() -> ModelSettings:
    bootstrap_environment()

    project_root = _PROJECT_ROOT
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

    preferred_answer_model = resolve_preferred_answer_model()
    default_aux_model = _get_env_value("OLLAMA_MODEL") or preferred_answer_model
    roles = RoleModels(
        answer=_get_env_value("ANSWER_MODEL") or preferred_answer_model,
        router=_get_env_value("ROUTER_MODEL") or default_aux_model,
        self_check=_get_env_value("SELF_CHECK_MODEL") or default_aux_model,
        rewrite=_get_env_value("QUERY_REWRITE_MODEL") or default_aux_model,
        compress=_get_env_value("COMPRESSOR_MODEL") or default_aux_model,
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
        temperature_answer=_env_float("ANSWER_TEMPERATURE", 0.4),
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
        query_rewrite_enabled=_env_bool("QUERY_REWRITE_ENABLED", True),
        query_rewrite_min_similarity=max(
            0.0,
            min(1.0, _env_float("QUERY_REWRITE_MIN_SIMILARITY", 0.85)),
        ),
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

    raw_backend = os.getenv("SAFETY_BACKEND", "nemo").strip().lower()
    safety_backend: SafetyBackend = "off" if raw_backend == "off" else "nemo"

    nemo_config_path_raw = os.getenv("NEMO_CONFIG_PATH", "").strip()
    nemo_config_path = Path(nemo_config_path_raw) if nemo_config_path_raw else None
    nemo_context_config_path_raw = os.getenv("NEMO_CONTEXT_CONFIG_PATH", "").strip()
    nemo_context_config_path = Path(nemo_context_config_path_raw) if nemo_context_config_path_raw else None
    nemo_output_config_path_raw = os.getenv("NEMO_OUTPUT_CONFIG_PATH", "").strip()
    nemo_output_config_path = Path(nemo_output_config_path_raw) if nemo_output_config_path_raw else None
    security_fallback_text = os.getenv(
        "SAFETY_SECURITY_FALLBACK_TEXT",
        "I cannot provide a safe, policy-compliant answer for this request. Please rephrase.",
    ).strip()

# ein objekt wird erstellt und zurückgegeben 
    safety = SafetyConfig(
        enabled=_env_bool("SAFETY_ENABLED", True),
        mode=safety_mode,
        min_groundedness=min_groundedness,
        block_pii=_env_bool("SAFETY_BLOCK_PII", True), # wenn true, dann werden pii-daten blockiert
        block_injection=_env_bool("SAFETY_BLOCK_INJECTION", True),
        fail_closed=_env_bool("SAFETY_FAIL_CLOSED", True), # wenn true, beim fehlschlagen oder zweifelhaften ergebnis, wird der fallback-text zurückgegeben
        fallback_text=os.getenv(
            "SAFETY_FALLBACK_TEXT",
            security_fallback_text,
        ).strip(),
        security_fallback_text=security_fallback_text,
        grounding_fallback_text=os.getenv(
            "SAFETY_GROUNDING_FALLBACK_TEXT",
            "I could not generate an answer that is sufficiently supported by the available documents.",
        ).strip(),
        pii_fallback_text=os.getenv(
            "SAFETY_PII_FALLBACK_TEXT",
            "I cannot include or expose personal or sensitive information in the response.",
        ).strip(),
        context_fallback_text=os.getenv(
            "SAFETY_CONTEXT_FALLBACK_TEXT",
            "The retrieved context contained information that could not be safely used for answering.",
        ).strip(),
        pii_allowed_emails=_env_csv("SAFETY_PII_ALLOWED_EMAILS"), # liste von erlaubten emails  
        pii_allowed_phone_numbers=_env_csv("SAFETY_PII_ALLOWED_PHONE_NUMBERS"), # liste von erlaubten telefonnummern
        pii_allowed_domains=_env_csv("SAFETY_PII_ALLOWED_DOMAINS"), # liste von erlaubten domains
        pii_allowed_phone_prefixes=_env_csv("SAFETY_PII_ALLOWED_PHONE_PREFIXES"), # liste von erlaubten telefonprefixen

        # Neu
        backend=safety_backend,
        nemo_config_path=nemo_config_path,
        nemo_context_config_path=nemo_context_config_path,
        nemo_output_config_path=nemo_output_config_path,
        nemo_input_enabled=_env_bool("NEMO_INPUT_ENABLED", True),
        nemo_context_enabled=_env_bool("NEMO_CONTEXT_ENABLED", True),
        nemo_output_enabled=_env_bool("NEMO_OUTPUT_ENABLED", True),
        nemo_enforce_input=_env_bool("NEMO_ENFORCE_INPUT", True),
        nemo_enforce_output=_env_bool("NEMO_ENFORCE_OUTPUT", True),
        nemo_runtime_timeout_seconds=_env_float("NEMO_RUNTIME_TIMEOUT_SECONDS", 20.0),
    )

    return ModelSettings(
        provider=provider,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        preferred_answer_model=preferred_answer_model,
        roles=roles,
        embedding=embedding,
        reranker=reranker,
        generation=generation,
        retrieval=retrieval,
        storage=storage,
        chunking=chunking,
        safety=safety,
    )
