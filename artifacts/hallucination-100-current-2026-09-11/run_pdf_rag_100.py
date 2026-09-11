"""Artifact-local 100-case wrapper for the existing PDF RAG benchmark runner."""

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[2]
dataset = ROOT / "artifacts" / "hallucination-100-current-2026-09-11" / "dataset_100.jsonl"
output = ROOT / "artifacts" / "hallucination-100-current-2026-09-11" / "runtime-sequential"

# The canonical runner currently enforces exactly 200 rows. Execute an artifact-local
# copy with only that cardinality assertion adjusted; its request and audit logic stay intact.
source_path = ROOT / "scripts" / "evaluation" / "run_pdf_rag_200.py"
source = source_path.read_text(encoding="utf-8")
source = source.replace(
    'if len(cases) != 200:\n        raise SystemExit(f"Expected 200 cases, got {len(cases)}")',
    'if len(cases) != 100:\n        raise SystemExit(f"Expected 100 cases, got {len(cases)}")',
)
sys.argv = [
    str(source_path),
    "--dataset", str(dataset),
    "--output-dir", str(output),
    "--workers", "1",
    "--timeout-seconds", "190",
]
exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
