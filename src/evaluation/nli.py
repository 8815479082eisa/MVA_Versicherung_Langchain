"""NLI abstraction layer for citation-oriented evaluation."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Optional


DEFAULT_TRUE_NLI_MODEL = "google/t5_xxl_true_nli_mixture"
DEFAULT_BART_MNLI_MODEL = "facebook/bart-large-mnli"


class NLIModel(ABC):
    """Abstract NLI model."""

    backend: str = "unknown"
    available: bool = True
    unavailable_reason: Optional[str] = None

    @abstractmethod
    def entail(self, premise: str, hypothesis: str) -> bool:
        """Return True when premise entails hypothesis."""


class UnavailableNLIModel(NLIModel):
    """Fallback model used when a backend cannot be initialized."""

    def __init__(self, backend: str, reason: str):
        self.backend = backend
        self.available = False
        self.unavailable_reason = reason

    def entail(self, premise: str, hypothesis: str) -> bool:
        del premise, hypothesis
        raise RuntimeError(
            f"NLI backend '{self.backend}' is unavailable: {self.unavailable_reason}"
        )


class SQLiteNLICache:
    """Persistent cache for NLI entailment calls."""

    def __init__(self, cache_path: str | os.PathLike):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.cache_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nli_cache (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[bool]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM nli_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return bool(row[0])

    def set(self, key: str, value: bool) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO nli_cache(key, value) VALUES (?, ?)",
                (key, 1 if value else 0),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _normalize_label(label: str) -> str:
    return (label or "").strip().lower()


class BartMNLIModel(NLIModel):
    """NLI backend using facebook/bart-large-mnli."""

    backend = "bart_mnli"

    def __init__(self, model_name: str = DEFAULT_BART_MNLI_MODEL):
        self.model_name = model_name
        self._classifier = None

    def _load(self):
        if self._classifier is not None:
            return self._classifier
        try:
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency `transformers` for bart_mnli backend."
            ) from exc
        self._classifier = pipeline(
            "text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
        )
        return self._classifier

    def entail(self, premise: str, hypothesis: str) -> bool:
        classifier = self._load()
        try:
            result = classifier(
                {"text": premise, "text_pair": hypothesis},
                truncation=True,
                max_length=512,
                top_k=None,
            )
        except TypeError:
            # Fallback path for older pipeline versions.
            result = classifier(
                f"{premise} </s></s> {hypothesis}",
                truncation=True,
                max_length=512,
            )

        if isinstance(result, list) and result and isinstance(result[0], list):
            result = result[0]

        if isinstance(result, list):
            best = max(result, key=lambda item: float(item.get("score", 0.0)))
            label = _normalize_label(str(best.get("label", "")))
            if label in {"label_2", "entailment"} or "entail" in label:
                return True
            return False

        label = _normalize_label(str(result.get("label", "")))
        return label in {"label_2", "entailment"} or "entail" in label


class TrueNLIModel(NLIModel):
    """NLI backend using google/t5_xxl_true_nli_mixture (paper-like)."""

    backend = "true_nli"

    def __init__(self, model_name: str = DEFAULT_TRUE_NLI_MODEL):
        self.model_name = model_name
        self._generator = None

    def _load(self):
        if self._generator is not None:
            return self._generator
        try:
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency `transformers` for true_nli backend."
            ) from exc
        self._generator = pipeline(
            "text2text-generation",
            model=self.model_name,
            tokenizer=self.model_name,
        )
        return self._generator

    def entail(self, premise: str, hypothesis: str) -> bool:
        generator = self._load()
        prompt = f"premise: {premise} hypothesis: {hypothesis}"
        result = generator(
            prompt,
            max_new_tokens=8,
            do_sample=False,
            return_full_text=False,
        )
        if not result:
            return False
        text = str(result[0].get("generated_text", "")).strip().lower()
        if text.startswith(("1", "true", "entail")):
            return True
        if text.startswith(("0", "false", "neutral", "contradiction")):
            return False
        return "entail" in text


class CachedNLIModel(NLIModel):
    """Wraps another NLI model with in-memory + sqlite caching."""

    def __init__(
        self,
        inner_model: NLIModel,
        cache_path: str | os.PathLike = ".nli_cache.sqlite",
        in_memory_cache_size: int = 10_000,
    ):
        self.inner_model = inner_model
        self.backend = inner_model.backend
        self.available = inner_model.available
        self.unavailable_reason = inner_model.unavailable_reason
        self._cache = SQLiteNLICache(cache_path) if cache_path else None
        self._entail_uncached = lru_cache(maxsize=in_memory_cache_size)(
            self._entail_uncached_impl
        )

    @staticmethod
    def _make_key(backend: str, premise: str, hypothesis: str) -> str:
        payload = f"{backend}\n{premise}\n{hypothesis}".encode("utf-8", errors="ignore")
        return hashlib.sha256(payload).hexdigest()

    def _entail_uncached_impl(self, premise: str, hypothesis: str) -> bool:
        return self.inner_model.entail(premise, hypothesis)

    def entail(self, premise: str, hypothesis: str) -> bool:
        if not self.available:
            return self.inner_model.entail(premise, hypothesis)

        key = self._make_key(self.backend, premise, hypothesis)
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        result = self._entail_uncached(premise, hypothesis)
        if self._cache is not None:
            self._cache.set(key, result)
        return result


def create_nli_model(
    backend: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> NLIModel:
    """
    Create an NLI model based on backend name.

    Supported backends:
    - true_nli  (google/t5_xxl_true_nli_mixture)
    - bart_mnli (facebook/bart-large-mnli)
    """
    selected = (backend or os.getenv("NLI_BACKEND", "bart_mnli")).strip().lower()
    cache_file = cache_path or os.getenv(
        "NLI_CACHE_PATH", "data/processed/caches/nli_cache.sqlite"
    )

    if selected not in {"true_nli", "bart_mnli"}:
        selected = "bart_mnli"

    try:
        # Graceful degradation when optional dependencies are not present.
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except Exception as dep_exc:
            return UnavailableNLIModel(backend=selected, reason=str(dep_exc))

        base: NLIModel
        if selected == "true_nli":
            base = TrueNLIModel()
        else:
            base = BartMNLIModel()
        return CachedNLIModel(base, cache_path=cache_file)
    except Exception as exc:
        return UnavailableNLIModel(backend=selected, reason=str(exc))
