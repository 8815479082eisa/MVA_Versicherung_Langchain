import unittest
from unittest.mock import AsyncMock

from src.mcp_clients.insurance_tools import MCPRuntimeError, PersistentMCPToolRuntime


class MCPToolRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_text_block_is_parsed_as_json(self):
        runtime = PersistentMCPToolRuntime()
        runtime._exit_stack = AsyncMock()
        tool = AsyncMock()
        tool.ainvoke.return_value = [
            {
                "type": "text",
                "text": '{"ok":true,"available":true,"found":false}',
            }
        ]
        runtime._tools = {"get_policy": tool}

        result = await runtime.invoke(
            "get_policy",
            {"policy_number": "TEST-KFZ-2026-1001"},
        )
        self.assertTrue(result["ok"])
        self.assertFalse(result["found"])

    async def test_non_json_text_is_rejected(self):
        runtime = PersistentMCPToolRuntime()
        runtime._exit_stack = AsyncMock()
        tool = AsyncMock()
        tool.ainvoke.return_value = [{"type": "text", "text": "not-json"}]
        runtime._tools = {"get_policy": tool}
        with self.assertRaises(MCPRuntimeError):
            await runtime.invoke(
                "get_policy",
                {"policy_number": "TEST-KFZ-2026-1001"},
            )


if __name__ == "__main__":
    unittest.main()
