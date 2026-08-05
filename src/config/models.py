from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - optional in some tooling contexts
    dotenv_values = None

Provider = Literal["ollama", "openai"]
SafetyMode = Literal["off", "monitor", "enforce"]
SafetyBackend = Literal["nemo", "off"]

_DEFAULT_PREFERRED_ANSWER_MODEL = "qwen3.5:4b"
_DEFAULT_OPENAI_ANSWER_MODEL = "gpt-4o-mini"
_DEFAULT_OPENAI_SELF_CHECK_MODEL = "gpt-4o-mini"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DOTENV_PATH = _PROJECT_ROOT / ".env"
_GROUNDEDNESS_CALIBRATION_PATH = _PROJECT_ROOT / "config" / "groundedness_calibration.json"
_GROUNDEDNESS_ALGORITHM_VERSION = "fact_aware_claim_support_v5"
_ENV_SOURCES: Dict[str, str] = {}
_DOTENV_VALUES: Dict[str, str] = {}
_DOTENV_CONFLICTS: Dict[str, Dict[str, str]] = {}
_SENSITIVE_ENV_NAME_MARKERS = (
    "API_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "AUTH",
)


def _safe_dotenv_conflicts() -> Dict[str, Dict[str, str]]:
    """Return diagnostics without persisting credentials from environment values."""

    sanitized: Dict[str, Dict[str, str]] = {}
    for name, values in _DOTENV_CONFLICTS.items():
        is_sensitive = any(marker in name.upper() for marker in _SENSITIVE_ENV_NAME_MARKERS)
        sanitized[name] = {
            source: "<redacted>" if is_sensitive else value
            for source, value in values.items()
        }
    return sanitized


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


def load_groundedness_calibration(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    calibration_path = path or Path(
        os.getenv(
            "SAFETY_GROUNDEDNESS_CALIBRATION_FILE",
            str(_GROUNDEDNESS_CALIBRATION_PATH),
        )
    )
    if not calibration_path.is_file():
        return None

    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        threshold = float(payload["selected_threshold"])
        dataset_size = int(payload["dataset_size"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None

    if payload.get("algorithm_version") != _GROUNDEDNESS_ALGORITHM_VERSION:
        return None
    if not 0.0 <= threshold <= 1.0 or dataset_size < 20:
        return None

    return {
        **payload,
        "selected_threshold": threshold,
        "dataset_size": dataset_size,
        "path": str(calibration_path.resolve()),
    }


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
    default_answer_model = (
        _DEFAULT_OPENAI_ANSWER_MODEL
        if current_settings.provider == "openai"
        else _DEFAULT_PREFERRED_ANSWER_MODEL
    )
    configured_answer_model_source, configured_answer_model_source_detail = _resolve_model_source(
        "ANSWER_MODEL",
        fallback_env_name="PREFERRED_ANSWER_MODEL",
        default_value=default_answer_model,
    )
    preferred_answer_model_source, preferred_answer_model_source_detail = _resolve_model_source(
        "PREFERRED_ANSWER_MODEL",
        default_value=default_answer_model,
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
        "ANSWER_PROVIDER": os.getenv("ANSWER_PROVIDER"),
        "ANSWER_MODEL": os.getenv("ANSWER_MODEL"),
        "PREFERRED_ANSWER_MODEL": os.getenv("PREFERRED_ANSWER_MODEL"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL"),
        "ROUTER_MODEL": os.getenv("ROUTER_MODEL"),
        "GUARDRAIL_MODEL": os.getenv("GUARDRAIL_MODEL"),
        "SELF_CHECK_ENABLED": os.getenv("SELF_CHECK_ENABLED"),
        "SELF_CHECK_PROVIDER": os.getenv("SELF_CHECK_PROVIDER"),
        "SELF_CHECK_MODEL": os.getenv("SELF_CHECK_MODEL"),
        "SAFETY_BACKEND": os.getenv("SAFETY_BACKEND"),
        "NEMO_ENFORCE_OUTPUT": os.getenv("NEMO_ENFORCE_OUTPUT"),
        "QUERY_REWRITE_ENABLED": os.getenv("QUERY_REWRITE_ENABLED"),
        "INSURANCEQA_EXACT_MATCH_SHORTCUT": os.getenv("INSURANCEQA_EXACT_MATCH_SHORTCUT"),
    }

    return {
        "raw_env": raw_env,
        "dotenv_path": str(_DOTENV_PATH) if _DOTENV_PATH.exists() else None,
        "dotenv_conflicts": _safe_dotenv_conflicts(),
        "answer_provider": current_settings.provider,
        "answer_provider_source": _env_source("ANSWER_PROVIDER"),
        "openai_api_key_configured": current_settings.openai_api_key_configured,
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
        "guardrail_model": current_settings.roles.guardrail,
        "router_model_source": router_model_source,
        "router_model_source_detail": router_model_source_detail,
        "self_check_enabled": current_settings.retrieval.self_check_enabled,
        "self_check_provider": current_settings.retrieval.self_check_provider,
        "self_check_model": current_settings.roles.self_check,
        "query_rewrite_enabled": current_settings.retrieval.query_rewrite_enabled,
        "llm_output_limits": {
            stage: current_settings.llm_runtime.output_limit(stage)
            for stage in (
                "router",
                "guardrail",
                "self_check",
                "query_rewrite",
                "compressor",
                "reranker",
                "answer",
            )
        },
        "llm_timeouts_seconds": {
            stage: current_settings.llm_runtime.timeout(stage)
            for stage in (
                "router",
                "guardrail",
                "self_check",
                "query_rewrite",
                "compressor",
                "reranker",
                "answer",
                "retrieval",
            )
        },
        "api_request_timeout_seconds": (
            current_settings.llm_runtime.api_request_timeout_seconds
        ),
        "llm_max_retries": current_settings.llm_runtime.max_retries,
        "retrieval_max_retries": current_settings.llm_runtime.retrieval_max_retries,
        "nemo_enforce_output": current_settings.safety.nemo_enforce_output,
        "safety_backend": current_settings.safety.backend,
        "safety_min_groundedness": current_settings.safety.min_groundedness,
        "safety_min_groundedness_source": current_settings.safety.min_groundedness_source,
        "groundedness_calibration_file": (
            str(current_settings.safety.groundedness_calibration_file)
            if current_settings.safety.groundedness_calibration_file
            else None
        ),
        "pipeline_user_type": current_settings.safety.user_type,
        "pipeline_authenticated": current_settings.safety.authenticated,
        "pipeline_access_mode": current_settings.safety.access_mode,
        "pipeline_channel": current_settings.safety.channel,
        "internal_caseworker_mode": current_settings.safety.internal_caseworker_mode,
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


def resolve_preferred_answer_model(provider: Provider = "ollama") -> str:
    raw = _get_env_value("PREFERRED_ANSWER_MODEL")
    if raw:
        return raw
    if provider == "openai":
        return _DEFAULT_OPENAI_ANSWER_MODEL
    return _DEFAULT_PREFERRED_ANSWER_MODEL


@dataclass(frozen=True)
class RoleModels:
    answer: str
    router: str
    guardrail: str
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
class LLMRuntimeConfig:
    max_tokens_router: int
    max_tokens_guardrail: int
    max_tokens_self_check: int
    max_tokens_query_rewrite: int
    max_tokens_compressor: int
    max_tokens_reranker: int
    max_tokens_answer: int
    timeout_router_seconds: float
    timeout_guardrail_seconds: float
    timeout_self_check_seconds: float
    timeout_query_rewrite_seconds: float
    timeout_compressor_seconds: float
    timeout_reranker_seconds: float
    timeout_answer_seconds: float
    retrieval_timeout_seconds: float
    api_request_timeout_seconds: float
    max_retries: int
    retrieval_max_retries: int

    def output_limit(self, stage: str) -> int:
        normalized = stage.strip().lower()
        mapping = {
            "router": self.max_tokens_router,
            "guardrail": self.max_tokens_guardrail,
            "self_check": self.max_tokens_self_check,
            "query_rewrite": self.max_tokens_query_rewrite,
            "compressor": self.max_tokens_compressor,
            "reranker": self.max_tokens_reranker,
            "answer": self.max_tokens_answer,
        }
        if normalized not in mapping:
            raise KeyError(f"Unknown LLM stage: {stage}")
        return mapping[normalized]

    def timeout(self, stage: str) -> float:
        normalized = stage.strip().lower()
        mapping = {
            "router": self.timeout_router_seconds,
            "guardrail": self.timeout_guardrail_seconds,
            "self_check": self.timeout_self_check_seconds,
            "query_rewrite": self.timeout_query_rewrite_seconds,
            "compressor": self.timeout_compressor_seconds,
            "reranker": self.timeout_reranker_seconds,
            "answer": self.timeout_answer_seconds,
            "retrieval": self.retrieval_timeout_seconds,
        }
        if normalized not in mapping:
            raise KeyError(f"Unknown runtime stage: {stage}")
        return mapping[normalized]


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    bm25_k: int
    vector_k: int
    max_self_check_retries: int
    self_check_enabled: bool
    answer_completeness_enabled: bool
    self_check_provider: Provider
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
    min_groundedness_source: str = "environment_or_default"
    groundedness_calibration_file: Optional[Path] = None
    user_type: str = "internal_insurance_caseworker"
    authenticated: bool = False
    access_mode: str = "read_only"
    channel: str = "internal"

    @property
    def internal_caseworker_mode(self) -> bool:
        return (
            self.user_type == "internal_insurance_caseworker"
            and self.authenticated
            and self.access_mode == "read_only"
            and self.channel == "internal"
        )


@dataclass(frozen=True)
class ModelSettings:
    provider: Provider
    openai_api_key_configured: bool
    ollama_base_url: str
    preferred_answer_model: str
    roles: RoleModels
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    generation: GenerationConfig
    llm_runtime: LLMRuntimeConfig
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

    # Only final answer generation may use OpenAI. All auxiliary roles stay local.
    raw_provider = (_get_env_value("ANSWER_PROVIDER") or "ollama").lower()
    provider: Provider = "openai" if raw_provider == "openai" else "ollama"

    preferred_answer_model = resolve_preferred_answer_model(provider)
    raw_self_check_provider = (_get_env_value("SELF_CHECK_PROVIDER") or "").lower()
    self_check_provider: Provider = (
        "openai" if raw_self_check_provider == "openai" else "ollama"
    )
    default_aux_model = _get_env_value("OLLAMA_MODEL") or _DEFAULT_PREFERRED_ANSWER_MODEL
    self_check_model_value = _get_env_value("SELF_CHECK_MODEL")
    guardrail_self_check_fallback = (
        self_check_model_value if self_check_provider != "openai" else None
    )
    roles = RoleModels(
        answer=_get_env_value("ANSWER_MODEL") or preferred_answer_model,
        router=_get_env_value("ROUTER_MODEL") or default_aux_model,
        guardrail=(
            _get_env_value("GUARDRAIL_MODEL")
            or guardrail_self_check_fallback
            or default_aux_model
        ),
        self_check=(
            self_check_model_value
            or (
                _DEFAULT_OPENAI_SELF_CHECK_MODEL
                if self_check_provider == "openai"
                else default_aux_model
            )
        ),
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

    def _positive_int(name: str, default: int) -> int:
        return max(1, _env_int(name, default))

    def _positive_float(name: str, default: float) -> float:
        return max(0.1, _env_float(name, default))

    llm_runtime = LLMRuntimeConfig(
        max_tokens_router=_positive_int("LLM_MAX_TOKENS_ROUTER", 8),
        max_tokens_guardrail=_positive_int("LLM_MAX_TOKENS_GUARDRAIL", 8),
        max_tokens_self_check=_positive_int("LLM_MAX_TOKENS_SELF_CHECK", 8),
        max_tokens_query_rewrite=_positive_int("LLM_MAX_TOKENS_QUERY_REWRITE", 64),
        max_tokens_compressor=_positive_int("LLM_MAX_TOKENS_COMPRESSOR", 384),
        max_tokens_reranker=_positive_int("LLM_MAX_TOKENS_RERANKER", 64),
        max_tokens_answer=_positive_int("LLM_MAX_TOKENS_ANSWER", 384),
        timeout_router_seconds=_positive_float("LLM_TIMEOUT_ROUTER_SECONDS", 20.0),
        timeout_guardrail_seconds=_positive_float("LLM_TIMEOUT_GUARDRAIL_SECONDS", 30.0),
        timeout_self_check_seconds=_positive_float("LLM_TIMEOUT_SELF_CHECK_SECONDS", 30.0),
        timeout_query_rewrite_seconds=_positive_float(
            "LLM_TIMEOUT_QUERY_REWRITE_SECONDS",
            45.0,
        ),
        timeout_compressor_seconds=_positive_float(
            "LLM_TIMEOUT_COMPRESSOR_SECONDS",
            60.0,
        ),
        timeout_reranker_seconds=_positive_float("LLM_TIMEOUT_RERANKER_SECONDS", 60.0),
        timeout_answer_seconds=_positive_float("LLM_TIMEOUT_ANSWER_SECONDS", 120.0),
        retrieval_timeout_seconds=_positive_float("RETRIEVAL_TIMEOUT_SECONDS", 60.0),
        api_request_timeout_seconds=_positive_float(
            "API_REQUEST_TIMEOUT_SECONDS",
            180.0,
        ),
        max_retries=max(0, _env_int("LLM_MAX_RETRIES", 1)),
        retrieval_max_retries=max(0, _env_int("RETRIEVAL_MAX_RETRIES", 1)),
    )

    retrieval = RetrievalConfig(
        top_k=_env_int("RETRIEVE_TOP_K", 8),
        bm25_k=_env_int("BM25_TOP_K", 5),
        vector_k=_env_int("VECTOR_TOP_K", 5),
        max_self_check_retries=_env_int("MAX_SELF_CHECK_RETRIES", 2),
        self_check_enabled=_env_bool("SELF_CHECK_ENABLED", False),
        answer_completeness_enabled=_env_bool(
            "ANSWER_COMPLETENESS_ENABLED",
            False,
        ),
        self_check_provider=self_check_provider,
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

    groundedness_calibration = load_groundedness_calibration()
    min_groundedness = (
        float(groundedness_calibration["selected_threshold"])
        if groundedness_calibration
        else _env_float("SAFETY_MIN_GROUNDEDNESS", 0.7)
    )
    min_groundedness_source = (
        "calibration_file"
        if groundedness_calibration
        else f"SAFETY_MIN_GROUNDEDNESS ({_env_source('SAFETY_MIN_GROUNDEDNESS')})"
    )
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
        nemo_runtime_timeout_seconds=llm_runtime.timeout_guardrail_seconds,
        min_groundedness_source=min_groundedness_source,
        groundedness_calibration_file=(
            Path(groundedness_calibration["path"])
            if groundedness_calibration
            else None
        ),
        user_type=os.getenv(
            "PIPELINE_USER_TYPE",
            "internal_insurance_caseworker",
        ).strip(),
        authenticated=_env_bool("PIPELINE_AUTHENTICATED", False),
        access_mode=os.getenv("PIPELINE_ACCESS_MODE", "read_only").strip(),
        channel=os.getenv("PIPELINE_CHANNEL", "internal").strip(),
    )

    return ModelSettings(
        provider=provider,
        openai_api_key_configured=_get_env_value("OPENAI_API_KEY") is not None,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        preferred_answer_model=preferred_answer_model,
        roles=roles,
        embedding=embedding,
        reranker=reranker,
        generation=generation,
        llm_runtime=llm_runtime,
        retrieval=retrieval,
        storage=storage,
        chunking=chunking,
        safety=safety,
    )
