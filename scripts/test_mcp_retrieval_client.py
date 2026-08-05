from __future__ import annotations

import asyncio
import json
import multiprocessing
import sys
from pathlib import Path
from typing import Any


QUESTION = "What should a consumer know before buying an annuity?"
TOP_K = 2
TOOL_TIMEOUT_SECONDS = 1800


def _print_json(label: str, payload: Any) -> None:
    print(f"{label}:")
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str), flush=True)


async def _async_get_tools(project_root: Path):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "insurance_retrieval": {
                "transport": "stdio",
                "command": str(project_root / ".venv" / "Scripts" / "python.exe"),
                "args": ["src/mcp_servers/retrieval_server.py"],
                "cwd": str(project_root),
            }
        }
    )
    return await client.get_tools()


async def _async_call_tool(project_root: Path, payload: dict[str, Any]) -> Any:
    tools = await _async_get_tools(project_root)
    retrieve_tool = next((tool for tool in tools if tool.name == "retrieve_documents"), None)
    if retrieve_tool is None:
        raise RuntimeError("TOOL_NOT_FOUND: retrieve_documents")
    return await retrieve_tool.ainvoke(payload)


def _call_tool_in_child(queue: multiprocessing.Queue, project_root: str, payload: dict[str, Any]) -> None:
    try:
        queue.put({"ok": True, "result": asyncio.run(_async_call_tool(Path(project_root), payload))})
    except Exception as exc:
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _invoke_with_timeout(project_root: Path, payload: dict[str, Any]) -> Any:
    queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_call_tool_in_child,
        args=(queue, str(project_root), payload),
    )
    process.start()
    process.join(TOOL_TIMEOUT_SECONDS)
    if process.is_alive():
        process.terminate()
        process.join(10)
        raise TimeoutError(f"retrieve_documents exceeded {TOOL_TIMEOUT_SECONDS} seconds")
    if queue.empty():
        raise RuntimeError(f"tool process exited with code {process.exitcode} without a result")
    message = queue.get()
    if not message.get("ok"):
        raise RuntimeError(message.get("error", "unknown tool-call error"))
    return message.get("result")


async def _main_async() -> int:
    try:
        import langchain_mcp_adapters.client  # noqa: F401
    except Exception as exc:
        print(f"DEPENDENCY_ERROR: failed to import langchain_mcp_adapters: {type(exc).__name__}: {exc}")
        return 1

    project_root = Path(__file__).resolve().parents[1]

    try:
        tools = await _async_get_tools(project_root)
    except Exception as exc:
        print(f"MCP_CLIENT_ERROR: failed to start or query MCP server: {type(exc).__name__}: {exc}")
        return 1

    tool_names = [tool.name for tool in tools]
    _print_json("TOOLS_FOUND", tool_names)

    retrieve_tool = next((tool for tool in tools if tool.name == "retrieve_documents"), None)
    if retrieve_tool is None:
        print("TOOL_NOT_FOUND: retrieve_documents")
        return 1

    try:
        result = _invoke_with_timeout(
            project_root,
            {"question": QUESTION, "top_k": TOP_K},
        )
    except TimeoutError:
        print(f"TOOL_CALL_TIMEOUT: retrieve_documents exceeded {TOOL_TIMEOUT_SECONDS} seconds")
        return 1
    except Exception as exc:
        print(f"TOOL_CALL_ERROR: retrieve_documents failed: {type(exc).__name__}: {exc}")
        return 1

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            print("TOOL_CALL_SUCCESS: True")
            print("TOOL_RESULT_RAW:")
            print(result[:1000])
            return 0

    documents = result.get("documents", []) if isinstance(result, dict) else []
    first_doc = documents[0] if documents else {}
    first_source = first_doc.get("source", {}) if isinstance(first_doc, dict) else {}
    first_content = first_doc.get("content", "") if isinstance(first_doc, dict) else ""

    print("TOOL_CALL_SUCCESS: True")
    print(f"DOCUMENT_COUNT: {result.get('document_count') if isinstance(result, dict) else None}")
    _print_json(
        "FIRST_DOCUMENT_SOURCE",
        {
            "source": first_source.get("source"),
            "source_type": first_source.get("source_type"),
            "page": first_source.get("page"),
        },
    )
    print("FIRST_DOCUMENT_CONTENT_300:")
    print(first_content[:300])
    return 0


def main() -> int:
    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())
