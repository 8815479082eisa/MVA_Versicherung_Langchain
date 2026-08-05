import logging
import os
import sys
import time
from pathlib import Path

_LOCAL_MODELS_ONLY = os.getenv("MCP_RETRIEVAL_LOCAL_MODELS_ONLY", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
_WARMUP_ENABLED = os.getenv("MCP_RETRIEVAL_WARMUP", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
if _LOCAL_MODELS_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.rag_service import initialize_retrieval_service, retrieve_documents_for_tool


mcp = FastMCP("Insurance Retrieval")


@mcp.tool()
async def retrieve_documents(question: str, top_k: int = 5) -> dict:
    """
    Retrieve relevant insurance documents using the existing hybrid retrieval pipeline.
    Returns relevant chunks with metadata and source information.
    """
    return retrieve_documents_for_tool(question=question, top_k=top_k)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger(__name__).info(
        "MCP retrieval server starting (local_models_only=%s, warmup=%s)",
        _LOCAL_MODELS_ONLY,
        _WARMUP_ENABLED,
    )
    if _WARMUP_ENABLED:
        warmup_started = time.perf_counter()
        initialize_retrieval_service(force_reindex=False, allow_reindex=False)
        logging.getLogger(__name__).info(
            "MCP retrieval warm-up completed in %.3fs",
            time.perf_counter() - warmup_started,
        )
    mcp.run(transport="stdio")
