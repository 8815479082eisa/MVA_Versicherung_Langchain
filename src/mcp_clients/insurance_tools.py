from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Iterable

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from src.integrations.espocrm_client import EspoCRMConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MCPRuntimeError(RuntimeError):
    pass


def build_mcp_connections() -> dict[str, dict[str, Any]]:
    """Return both independent tool sources without starting either process."""

    crm_config = EspoCRMConfig.from_environment()
    return {
        "insurance_retrieval": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["src/mcp_servers/retrieval_server.py"],
            "cwd": str(PROJECT_ROOT),
            "env": {
                "MCP_RETRIEVAL_LOCAL_MODELS_ONLY": os.getenv(
                    "MCP_RETRIEVAL_LOCAL_MODELS_ONLY",
                    "true",
                ),
                "MCP_RETRIEVAL_WARMUP": os.getenv(
                    "MCP_RETRIEVAL_WARMUP",
                    "true",
                ),
            },
        },
        "insurance_crm": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["src/mcp_servers/crm_server.py"],
            "cwd": str(PROJECT_ROOT),
            "env": {
                "CRM_ENABLED": "true" if crm_config.enabled else "false",
                "ESPOCRM_BASE_URL": crm_config.base_url,
                "ESPOCRM_API_KEY": crm_config.api_key,
                "ESPOCRM_REQUEST_TIMEOUT_SECONDS": str(
                    crm_config.request_timeout_seconds
                ),
                "ESPOCRM_MAX_PAGES": str(crm_config.max_pages),
                "ESPOCRM_PAGE_SIZE": str(crm_config.page_size),
            },
        },
    }


class PersistentMCPToolRuntime:
    """Application-lifetime MCP sessions and LangChain tool handles."""

    def __init__(
        self,
        connections: dict[str, dict[str, Any]] | None = None,
    ):
        self.connections = connections or build_mcp_connections()
        self._client: MultiServerMCPClient | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools: dict[str, BaseTool] = {}
        self._server_tools: dict[str, tuple[str, ...]] = {}
        self._lifecycle_lock = asyncio.Lock()

    @property
    def started(self) -> bool:
        return self._exit_stack is not None

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    @property
    def server_tools(self) -> dict[str, tuple[str, ...]]:
        return dict(self._server_tools)

    async def start(self, server_names: Iterable[str] | None = None) -> None:
        async with self._lifecycle_lock:
            if self._exit_stack is not None:
                return

            selected = tuple(server_names or self.connections.keys())
            unknown = sorted(set(selected).difference(self.connections))
            if unknown:
                raise MCPRuntimeError(f"Unknown MCP servers: {unknown}")

            client = MultiServerMCPClient(self.connections, handle_tool_errors=True)
            stack = AsyncExitStack()
            tools: dict[str, BaseTool] = {}
            server_tools: dict[str, tuple[str, ...]] = {}
            try:
                await stack.__aenter__()
                for server_name in selected:
                    session = await stack.enter_async_context(
                        client.session(server_name)
                    )
                    loaded = await load_mcp_tools(
                        session,
                        server_name=server_name,
                        handle_tool_errors=True,
                    )
                    names: list[str] = []
                    for tool in loaded:
                        if tool.name in tools:
                            raise MCPRuntimeError(
                                f"Duplicate MCP tool name: {tool.name}"
                            )
                        tools[tool.name] = tool
                        names.append(tool.name)
                    server_tools[server_name] = tuple(sorted(names))
            except Exception:
                await stack.aclose()
                raise

            self._client = client
            self._exit_stack = stack
            self._tools = tools
            self._server_tools = server_tools

    async def close(self) -> None:
        async with self._lifecycle_lock:
            stack, self._exit_stack = self._exit_stack, None
            self._client = None
            self._tools = {}
            self._server_tools = {}
        if stack is not None:
            await stack.aclose()

    async def invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if self._exit_stack is None:
            raise MCPRuntimeError("The MCP tool runtime has not been started.")
        tool = self._tools.get(tool_name)
        if tool is None:
            raise MCPRuntimeError(f"MCP tool is not registered: {tool_name}")

        result = await tool.ainvoke(arguments)
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            return _parse_json_tool_text(tool_name, result)
        if (
            isinstance(result, list)
            and len(result) == 1
            and isinstance(result[0], dict)
            and result[0].get("type") == "text"
            and isinstance(result[0].get("text"), str)
        ):
            return _parse_json_tool_text(tool_name, result[0]["text"])
        raise MCPRuntimeError(
            f"MCP tool {tool_name} returned an unexpected response type."
        )


_runtime: PersistentMCPToolRuntime | None = None


def get_mcp_runtime() -> PersistentMCPToolRuntime:
    global _runtime
    if _runtime is None:
        _runtime = PersistentMCPToolRuntime()
    return _runtime


def _parse_json_tool_text(tool_name: str, value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise MCPRuntimeError(
            f"MCP tool {tool_name} returned non-JSON text."
        ) from exc
    if not isinstance(parsed, dict):
        raise MCPRuntimeError(
            f"MCP tool {tool_name} returned JSON with an unexpected structure."
        )
    return parsed
