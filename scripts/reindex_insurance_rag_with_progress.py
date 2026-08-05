from __future__ import annotations

import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def log(message: str = "") -> None:
    print(message, flush=True)


def git_status_short() -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return (result.stdout + result.stderr).strip()


def collection_count(chromadb_module: Any, persist_directory: Path, collection_name: str) -> str:
    try:
        client = chromadb_module.PersistentClient(path=str(persist_directory))
        return str(client.get_collection(collection_name).count())
    except Exception as exc:
        return f"ERROR {type(exc).__name__}: {exc}"


def main() -> int:
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = reports_dir / f"reindex_insurance_rag_{timestamp}.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        start = time.perf_counter()

        try:
            import chromadb
            from src.api import rag_service as rs

            persist_directory = rs.SETTINGS.storage.chroma_persist_directory
            collection_name = rs.SETTINGS.storage.collection_name

            log("Insurance RAG Reindex With Progress")
            log(f"Started: {datetime.now().isoformat(timespec='seconds')}")
            log(f"Repository: {REPO_ROOT}")
            log(f"Log file: {log_path}")
            log("")

            log("== Preflight ==")
            log("git status --short:")
            status = git_status_short()
            log(status if status else "<clean>")
            log(f"Chroma persist directory: {persist_directory}")
            log(
                "insurance_rag_collection count before: "
                + collection_count(chromadb, persist_directory, collection_name)
            )
            log(
                "insuranceqa_collection count before: "
                + collection_count(chromadb, persist_directory, "insuranceqa_collection")
            )

            source_files = rs.get_source_files(rs.PDF_DIRECTORY)
            extension_counts = Counter(Path(file_path).suffix.lower() for file_path in source_files)
            log(f"Source file count: {len(source_files)}")
            log(f"Source extension distribution: {dict(sorted(extension_counts.items()))}")
            log("")

            log("== Reindex ==")
            log("Calling initialize_pipeline(force_reindex=True)")
            components = rs.initialize_pipeline(force_reindex=True)
            elapsed = time.perf_counter() - start
            log("")

            log("== Result ==")
            log("Status: SUCCESS")
            log(f"Elapsed seconds: {elapsed:.1f}")
            log(f"Component keys: {sorted(components.keys())}")
            log(f"Hybrid retriever available: {callable(components.get('hybrid_retriever'))}")
            log(
                "insurance_rag_collection count after: "
                + collection_count(chromadb, persist_directory, collection_name)
            )
            log(
                "insuranceqa_collection count after: "
                + collection_count(chromadb, persist_directory, "insuranceqa_collection")
            )
            log(f"Finished: {datetime.now().isoformat(timespec='seconds')}")
            return 0
        except Exception:
            elapsed = time.perf_counter() - start
            log("")
            log("== Result ==")
            log("Status: ERROR")
            log(f"Elapsed seconds: {elapsed:.1f}")
            traceback.print_exc()
            log(f"Finished: {datetime.now().isoformat(timespec='seconds')}")
            return 1
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
