"""
InsuranceQA ingestion pipeline.

Builds a separate ChromaDB collection ("insuranceqa_collection") containing Q/A pairs
as Documents so it can be used later as an additional retriever.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional

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
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

logger = logging.getLogger(__name__)


INSURANCEQA_COLLECTION_NAME = "insuranceqa_collection"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _require(dep, package: str) -> None:
    if dep is None:
        raise RuntimeError(
            f"Missing optional dependency '{package}'. "
            "Install project requirements to use this feature."
        )


def _default_jsonl_path() -> Path:
    # Keep this aligned with existing repo paths; do not hardcode an absolute path.
    from src.api import rag_service

    return rag_service.SETTINGS.storage.benchmark_root / "qa" / "insuranceqa" / "data_insuranceqa_1000.jsonl"


def _load_local_jsonl(path: Path, max_rows: Optional[int] = None) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _try_load_hf_dataset(split: str = "train", max_rows: Optional[int] = None) -> Optional[List[dict]]:
    try:
        from datasets import load_dataset
    except Exception:
        return None

    ds = load_dataset("deccan-ai/insuranceQA-v2", split=split)
    rows: List[dict] = []
    for item in ds:
        # Common field names across extracted JSONLs.
        q = item.get("question") or item.get("query") or item.get("q")
        a = item.get("answer") or item.get("response") or item.get("a")
        if not q or not a:
            continue
        rows.append({"question": q, "answer": a})
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def _rows_to_documents(rows: Iterable[dict]) -> List[Document]:
    docs: List[Document] = []
    for idx, row in enumerate(rows):
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        if not q or not a:
            continue
        docs.append(
            Document(
                page_content=f"Question: {q}\nAnswer: {a}",
                metadata={
                    "source": "insuranceqa",
                    "question_id": str(row.get("question_id") or row.get("id") or idx),
                    "dataset": "InsuranceQA",
                },
            )
        )
    return docs


def _split_documents(docs: List[Document]) -> List[Document]:
    from src.api import rag_service

    _require(RecursiveCharacterTextSplitter, "langchain-text-splitters")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=rag_service.SETTINGS.chunking.chunk_size,
        chunk_overlap=rag_service.SETTINGS.chunking.chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def build_insuranceqa_index(
    force_reindex: bool = False,
    jsonl_path: Optional[str] = None,
    batch_size: int = 500,
    max_rows: Optional[int] = None,
    progress: bool = False,
    log_every: int = 1,
) -> None:
    """
    Build (or update) the InsuranceQA Chroma collection.

    - Uses SETTINGS.storage.chroma_persist_directory for persistence.
    - Writes to a separate collection name so it never overwrites the document collection.
    """
    from src.api import rag_service

    _require(chromadb, "chromadb")
    _require(Chroma, "langchain-chroma")

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if log_every <= 0:
        raise ValueError("log_every must be > 0")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be > 0")

    total_start = time.perf_counter()
    path = Path(jsonl_path) if jsonl_path else _default_jsonl_path()
    rows: Optional[List[dict]] = None

    logger.info("[1/5] Load dataset started ...")
    stage_start = time.perf_counter()
    if path.exists():
        logger.info("Loading InsuranceQA JSONL: %s", path)
        rows = _load_local_jsonl(path, max_rows=max_rows)
    else:
        logger.warning("InsuranceQA JSONL not found at %s, trying HuggingFace dataset...", path)
        rows = _try_load_hf_dataset(
            split=os.getenv("INSURANCEQA_HF_SPLIT", "train"),
            max_rows=max_rows,
        )

    if not rows:
        logger.error("No InsuranceQA data available (missing JSONL and HuggingFace datasets not available).")
        return
    logger.info("[1/5] Load dataset done in %.2fs | rows=%d", time.perf_counter() - stage_start, len(rows))

    logger.info("[2/5] Map to Documents started ...")
    stage_start = time.perf_counter()
    base_docs = _rows_to_documents(rows)
    if not base_docs:
        logger.error("InsuranceQA dataset loaded but produced 0 documents.")
        return
    logger.info("[2/5] Map to Documents done in %.2fs | documents=%d",
                time.perf_counter() - stage_start, len(base_docs))

    logger.info("[3/5] Chunking started ...")
    stage_start = time.perf_counter()
    chunks = _split_documents(base_docs)
    logger.info("[3/5] Chunking done in %.2fs | chunks=%d", time.perf_counter() - stage_start, len(chunks))
    if not chunks:
        logger.error("Chunking produced 0 chunks; aborting index build.")
        return

    try:
        embeddings = rag_service.initialize_embeddings()
    except Exception as exc:
        logger.error(
            "Failed to initialize embeddings (%s). If you are using HuggingFace embeddings, "
            "ensure 'sentence-transformers' is installed.",
            exc,
        )
        return
    client = chromadb.PersistentClient(path=str(rag_service.SETTINGS.storage.chroma_persist_directory))
    collection = client.get_or_create_collection(name=INSURANCEQA_COLLECTION_NAME)

    if force_reindex and collection.count() > 0:
        logger.info("Force reindex enabled. Dropping existing collection '%s'.", INSURANCEQA_COLLECTION_NAME)
        client.delete_collection(name=INSURANCEQA_COLLECTION_NAME)
        client.create_collection(name=INSURANCEQA_COLLECTION_NAME)
        collection = client.get_or_create_collection(name=INSURANCEQA_COLLECTION_NAME)

    vector_store = Chroma(
        client=client,
        collection_name=INSURANCEQA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(rag_service.SETTINGS.storage.chroma_persist_directory),
    )

    if collection.count() > 0 and not force_reindex:
        logger.info("Collection '%s' already has %d items. Skipping rebuild (use --force-reindex).",
                    INSURANCEQA_COLLECTION_NAME, collection.count())
        return

    logger.info("[4/5] Indexing into Chroma started ...")
    logger.info(
        "Index target: collection='%s' | persist_dir='%s' | batch_size=%d",
        INSURANCEQA_COLLECTION_NAME,
        rag_service.SETTINGS.storage.chroma_persist_directory,
        batch_size,
    )
    if progress:
        logger.info(
            "If you don't see live progress, run: python -u -m src.data.insuranceqa_ingestion ..."
        )
    if progress and tqdm is None:
        logger.warning(
            "--progress was enabled, but tqdm is not installed. Falling back to log-only progress."
        )

    total_chunks = len(chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size
    stage_start = time.perf_counter()
    pbar = None
    if progress and tqdm is not None:
        pbar = tqdm(total=total_chunks, desc="Indexing", unit="chunk", dynamic_ncols=True)

    try:
        for batch_index, start_idx in enumerate(range(0, total_chunks, batch_size), start=1):
            batch = chunks[start_idx:start_idx + batch_size]
            vector_store.add_documents(documents=batch)
            done = min(start_idx + len(batch), total_chunks)
            pct = (done / total_chunks) * 100.0 if total_chunks else 100.0
            if pbar is not None:
                pbar.update(len(batch))
            if (batch_index % log_every == 0) or (done == total_chunks):
                logger.info(
                    "[4/5] Indexed %d/%d chunks (%.1f%%) | batch=%d/%d | elapsed=%.2fs",
                    done,
                    total_chunks,
                    pct,
                    batch_index,
                    total_batches,
                    time.perf_counter() - stage_start,
                )
    finally:
        if pbar is not None:
            pbar.close()

    indexing_elapsed = time.perf_counter() - stage_start
    logger.info("[4/5] Indexing into Chroma done in %.2fs", indexing_elapsed)

    logger.info("[5/5] Done / Summary")
    embedding_model = getattr(rag_service.SETTINGS.embedding, "model", "unknown")
    embedding_device = getattr(rag_service.SETTINGS.embedding, "device", "unknown")
    logger.info(
        "Summary | rows=%d | documents=%d | chunks=%d | embedding_model=%s | device=%s | collection=%s | persist_dir=%s | total_time=%.2fs",
        len(rows),
        len(base_docs),
        len(chunks),
        embedding_model,
        embedding_device,
        INSURANCEQA_COLLECTION_NAME,
        rag_service.SETTINGS.storage.chroma_persist_directory,
        time.perf_counter() - total_start,
    )


def get_insuranceqa_retriever(embeddings=None):
    """
    Load the InsuranceQA collection and return a retriever.
    """
    from src.api import rag_service

    _require(chromadb, "chromadb")
    _require(Chroma, "langchain-chroma")
    try:
        embeddings = embeddings or rag_service.initialize_embeddings()
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize embeddings for InsuranceQA retriever. "
            "If you are using HuggingFace embeddings, install 'sentence-transformers'."
        ) from exc
    client = chromadb.PersistentClient(path=str(rag_service.SETTINGS.storage.chroma_persist_directory))

    vector_store = Chroma(
        client=client,
        collection_name=INSURANCEQA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(rag_service.SETTINGS.storage.chroma_persist_directory),
    )
    return vector_store.as_retriever(search_kwargs={"k": rag_service.SETTINGS.retrieval.top_k})


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the InsuranceQA ChromaDB index.")
    parser.add_argument("--force-reindex", action="store_true", help="Drop and rebuild the InsuranceQA collection.")
    parser.add_argument("--jsonl", default=None, help="Path to local InsuranceQA JSONL (question/answer per line).")
    parser.add_argument("--progress", action="store_true", help="Enable live progress output.")
    parser.add_argument("--batch-size", type=int, default=500, help="Number of chunks to index per batch.")
    parser.add_argument("--max-rows", type=int, default=None, help="Load only the first N rows for quick tests.")
    parser.add_argument("--log-every", type=int, default=1, help="Log every N indexed batches.")
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be > 0")
    if args.log_every <= 0:
        parser.error("--log-every must be > 0")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    build_insuranceqa_index(
        force_reindex=bool(args.force_reindex),
        jsonl_path=args.jsonl,
        batch_size=int(args.batch_size),
        max_rows=args.max_rows,
        progress=bool(args.progress),
        log_every=int(args.log_every),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
