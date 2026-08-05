from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, TypeAlias

from src.core.llm_runtime import (
    GuardrailInvalidOutputError,
    LLMStageTimeoutError,
)

if TYPE_CHECKING:
    from nemoguardrails import LLMRails as LLMRailsType
else:
    LLMRailsType: TypeAlias = Any

try:
    from nemoguardrails import LLMRails, RailsConfig
except Exception:  # pragma: no cover - exercised in dependency-light test envs
    LLMRails = None
    RailsConfig = None

try:
    from config.models import SafetyConfig, load_model_settings
    from guardrails.types import StageResult, TraceEntry
except Exception:
    from src.config.models import SafetyConfig, load_model_settings
    from src.guardrails.types import StageResult, TraceEntry

logger = logging.getLogger(__name__)
_NEMO_GENERATION_GATE = threading.BoundedSemaphore(value=1)


def _extract_message_content(response: Any) -> str:
    if hasattr(response, "response"):
        messages = getattr(response, "response", None) or []
        if isinstance(messages, Mapping):
            return str(messages.get("content", ""))
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
            if messages and isinstance(messages[-1], Mapping):
                return str(messages[-1].get("content", ""))
    if isinstance(response, dict):
        return str(response.get("content", ""))
    return str(response or "")


def _extract_output_data(response: Any) -> dict[str, Any]:
    data = getattr(response, "output_data", None)
    if isinstance(data, dict):
        return dict(data)
    if isinstance(response, Mapping):
        response_output_data = response.get("output_data")
        if isinstance(response_output_data, Mapping):
            return dict(response_output_data)
    return {}


def _normalize_messages(messages: Any) -> list[dict[str, Any]]:
    if messages is None:
        return []
    if isinstance(messages, Mapping):
        return [dict(messages)]
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        normalized: list[dict[str, Any]] = []
        for item in messages:
            if not isinstance(item, Mapping):
                raise TypeError("NeMo messages must be mappings with 'role'/'content' fields.")
            normalized.append(dict(item))
        return normalized
    raise TypeError("NeMo messages must be a mapping or a sequence of mappings.")


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [str(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return [str(value)]


def _coerce_score_dict(value: Any) -> dict[str, float]:
    if isinstance(value, Mapping):
        payload: dict[str, float] = {}
        for key, raw_value in value.items():
            try:
                payload[str(key)] = float(raw_value)
            except (TypeError, ValueError):
                continue
        return payload
    return {}


def _safe_colang_history(rails: LLMRailsType) -> tuple[str | None, str | None]:
    try:
        history = rails.explain().colang_history
    except Exception as exc:  # pragma: no cover - defensive hardening for runtime diagnostics
        return None, f"{type(exc).__name__}: {exc}"

    if history is None:
        return None, None
    if isinstance(history, str):
        return history, None
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        return "\n".join(str(item) for item in history), None
    return str(history), None


def _as_stage_trace(stage: str, rail: str, action: str, reasons: list[str], details: dict[str, Any]) -> list[TraceEntry]:
    return [
        TraceEntry.now(
            stage=stage,  # type: ignore[arg-type]
            rail=rail,
            decision="allow" if action == "allow" else "deny",
            action=action,  # type: ignore[arg-type]
            reasons=reasons,
            details=details,
        )
    ]


def _serialize_docs(docs: Iterable[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for doc in docs or []:
        if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
            payload.append(
                {
                    "page_content": str(getattr(doc, "page_content", "") or ""),
                    "metadata": dict(getattr(doc, "metadata", {}) or {}),
                }
            )
        elif isinstance(doc, dict):
            payload.append(
                {
                    "page_content": str(doc.get("page_content", "")),
                    "metadata": dict(doc.get("metadata", {}) or {}),
                }
            )
        else:
            payload.append({"page_content": str(doc), "metadata": {}})
    return payload


def _build_main_llm_from_settings() -> Any:
    """Create the main LLM for NeMo runtime to avoid constructor warnings."""
    settings = load_model_settings()
    model_name = settings.roles.guardrail or settings.roles.self_check
    timeout_seconds = settings.llm_runtime.timeout_guardrail_seconds
    kwargs: dict[str, Any] = {
        "model": model_name,
        "base_url": settings.ollama_base_url,
        "temperature": settings.generation.temperature_aux,
        "num_predict": settings.llm_runtime.max_tokens_guardrail,
        "client_kwargs": {"timeout": timeout_seconds},
        "async_client_kwargs": {"timeout": timeout_seconds},
        "sync_client_kwargs": {"timeout": timeout_seconds},
    }

    try:
        from langchain_ollama import ChatOllama  # type: ignore
    except Exception:
        try:
            from langchain_community.chat_models import ChatOllama  # type: ignore
        except Exception:
            return None

    try:
        return ChatOllama(**kwargs)
    except Exception:
        return None


class OfficialNemoGuardrailsRuntime:
    """Stage-based wrapper around the official `nemoguardrails.LLMRails` runtime."""

    def __init__(self, config: SafetyConfig):
        if LLMRails is None or RailsConfig is None:
            raise ImportError("nemoguardrails is not installed.")
        self.config = config
        self._input_rails = self._load_rails(self._input_config_path())
        self._context_rails = self._load_rails(self._context_config_path()) if config.nemo_context_enabled else None
        self._output_rails = self._load_rails(self._output_config_path()) if config.nemo_output_enabled else None

    def _input_config_path(self) -> Path:
        if self.config.nemo_config_path is None:
            raise ValueError("NEMO_CONFIG_PATH is not set.")
        return Path(self.config.nemo_config_path)

    def _context_config_path(self) -> Path:
        if self.config.nemo_context_config_path is not None:
            return Path(self.config.nemo_context_config_path)
        if self.config.nemo_config_path is None:
            raise ValueError("NEMO_CONTEXT_CONFIG_PATH is not set.")
        return self._input_config_path().with_name(f"{self._input_config_path().name}_context")

    def _output_config_path(self) -> Path:
        if self.config.nemo_output_config_path is not None:
            return Path(self.config.nemo_output_config_path)
        if self.config.nemo_config_path is None:
            raise ValueError("NEMO_OUTPUT_CONFIG_PATH is not set.")
        return self._input_config_path().with_name(f"{self._input_config_path().name}_output")

    def _load_rails(self, config_path: Path) -> LLMRailsType:
        if not config_path.exists():
            raise ValueError(f"NeMo config path does not exist: {config_path}")
        rails_config = RailsConfig.from_path(str(config_path))
        main_llm = _build_main_llm_from_settings()
        rails = LLMRails(rails_config, llm=main_llm) if main_llm is not None else LLMRails(rails_config)
        rails.register_action_param("project_safety_config", self.config)
        return rails

    @staticmethod
    def _generate(
        rails: LLMRailsType,
        *,
        messages: Any,
        options: Optional[dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Any:
        normalized_messages = _normalize_messages(messages)
        result_box: dict[str, Any] = {}
        error_box: dict[str, Exception] = {}
        bounded_timeout = max(0.1, float(timeout_seconds or 30.0))
        if not _NEMO_GENERATION_GATE.acquire(timeout=bounded_timeout):
            raise LLMStageTimeoutError("guardrail")
        caller_context = contextvars.copy_context()

        def _runner() -> None:
            try:
                result_box["value"] = caller_context.run(
                    asyncio.run,
                    rails.generate_async(
                        messages=normalized_messages,
                        options=options,
                    ),
                )
            except Exception as exc:  # pragma: no cover - surfaced to caller
                error_box["error"] = exc
            finally:
                _NEMO_GENERATION_GATE.release()

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join(timeout=bounded_timeout)

        if thread.is_alive():
            raise LLMStageTimeoutError("guardrail")

        if error_box:
            raise error_box["error"]
        return result_box.get("value")

    def run_pre_query(self, query: str, chat_history=None) -> StageResult:
        del chat_history
        logger.debug("NEMO pre_query start payload_type=%s", type(query).__name__)
        response = self._generate(
            self._input_rails,
            messages=[{"role": "user", "content": query}],
            options={"rails": ["input"], "output_vars": True},
            timeout_seconds=self.config.nemo_runtime_timeout_seconds,
        )
        output_data = _extract_output_data(response)
        action = str(output_data.get("guardrails_input_action", ""))
        if action not in {"allow", "block", "fallback", "redact"}:
            raise GuardrailInvalidOutputError("guardrail")
        reasons = _coerce_str_list(output_data.get("guardrails_input_reasons", []))
        scores = _coerce_score_dict(output_data.get("guardrails_input_scores", {}))
        content = _extract_message_content(response)
        rail = str(output_data.get("triggered_input_rail", "official_nemo_input"))
        colang_history, colang_history_error = _safe_colang_history(self._input_rails)

        details = {
            "stage": "pre_query",
            "nemo_runtime_kind": "official_llmrails",
            "nemo_config_path": str(self._input_config_path()),
            "sentinel_response": content,
            "colang_history": colang_history,
            "output_data": output_data,
            "query_changed": False,
            "answer_changed": False,
        }
        if colang_history_error is not None:
            details["colang_history_error"] = colang_history_error

        return StageResult(
            stage="pre_query",
            allow=action == "allow",
            action=action,  # type: ignore[arg-type]
            query=query,
            answer="",
            reasons=reasons,
            scores=scores,
            details=details,
            trace=_as_stage_trace("pre_query", rail, action, reasons, details),
            blocked_by=None if action == "allow" else rail,
        )

    def run_context(self, query: str, docs: list[Any]) -> StageResult:
        if self._context_rails is None:
            return StageResult(stage="context", allow=True, action="allow", query=query, answer="")

        logger.debug("NEMO context stage start docs_count=%s", len(docs))
        response = self._generate(
            self._context_rails,
            messages=[
                {"role": "context", "content": {"context_docs": _serialize_docs(docs)}},
                {"role": "user", "content": query or ""},
            ],
            options={"rails": ["input"], "output_vars": True},
            timeout_seconds=self.config.nemo_runtime_timeout_seconds,
        )
        output_data = _extract_output_data(response)
        action = str(output_data.get("guardrails_context_action", ""))
        if action not in {"allow", "block", "fallback", "redact"}:
            raise GuardrailInvalidOutputError("guardrail")
        reasons = _coerce_str_list(output_data.get("guardrails_context_reasons", []))
        scores = _coerce_score_dict(output_data.get("guardrails_context_scores", {}))
        content = _extract_message_content(response)
        rail = str(output_data.get("triggered_input_rail", "official_nemo_context"))
        colang_history, colang_history_error = _safe_colang_history(self._context_rails)

        details = {
            "stage": "context",
            "nemo_runtime_kind": "official_llmrails",
            "nemo_config_path": str(self._context_config_path()),
            "sentinel_response": content,
            "colang_history": colang_history,
            "output_data": output_data,
            "sanitized_docs": output_data.get("guardrails_context_sanitized_docs"),
            "query_changed": False,
            "answer_changed": False,
        }
        if colang_history_error is not None:
            details["colang_history_error"] = colang_history_error

        return StageResult(
            stage="context",
            allow=action == "allow",
            action=action,  # type: ignore[arg-type]
            query=query,
            answer="",
            reasons=reasons,
            scores=scores,
            details=details,
            trace=_as_stage_trace("context", rail, action, reasons, details),
            blocked_by=None if action == "allow" else rail,
        )

    def run_post_generation(self, query: str, answer: str, docs: list[Any]) -> StageResult:
        if self._output_rails is None:
            return StageResult(stage="post_generation", allow=True, action="allow", query=query, answer=answer)

        logger.debug("NEMO post_generation start docs_count=%s", len(docs))
        response = self._generate(
            self._output_rails,
            messages=[
                {
                    "role": "context",
                    "content": {
                        "llm_output": answer,
                        "context_docs": _serialize_docs(docs),
                    },
                },
                {"role": "user", "content": query},
            ],
            options={"output_vars": True},
            timeout_seconds=self.config.nemo_runtime_timeout_seconds,
        )
        output_data = _extract_output_data(response)
        action = str(output_data.get("guardrails_output_action", ""))
        if action not in {"allow", "block", "fallback", "redact"}:
            raise GuardrailInvalidOutputError("guardrail")
        reasons = _coerce_str_list(output_data.get("guardrails_output_reasons", []))
        scores = _coerce_score_dict(output_data.get("guardrails_output_scores", {}))
        content = _extract_message_content(response)
        rail = str(output_data.get("triggered_output_rail", "official_nemo_output"))
        sanitized_answer = str(output_data.get("guardrails_output_sanitized_answer", answer))
        colang_history, colang_history_error = _safe_colang_history(self._output_rails)

        details = {
            "stage": "post_generation",
            "nemo_runtime_kind": "official_llmrails",
            "nemo_config_path": str(self._output_config_path()),
            "sentinel_response": content,
            "colang_history": colang_history,
            "output_data": output_data,
            "query_changed": False,
            "answer_changed": action == "redact" and sanitized_answer != answer,
        }
        action_details = output_data.get("guardrails_output_details")
        if isinstance(action_details, Mapping):
            details["action_details"] = dict(action_details)
            groundedness_details = action_details.get("groundedness")
            if isinstance(groundedness_details, Mapping):
                details["groundedness"] = dict(groundedness_details)
        if colang_history_error is not None:
            details["colang_history_error"] = colang_history_error

        return StageResult(
            stage="post_generation",
            allow=action == "allow",
            action=action,  # type: ignore[arg-type]
            query=query,
            answer=sanitized_answer if action == "redact" else answer,
            reasons=reasons,
            scores=scores,
            details=details,
            trace=_as_stage_trace("post_generation", rail, action, reasons, details),
            blocked_by=None if action == "allow" else rail,
        )


def create_nemo_official_runtime(config: SafetyConfig) -> OfficialNemoGuardrailsRuntime:
    return OfficialNemoGuardrailsRuntime(config)
