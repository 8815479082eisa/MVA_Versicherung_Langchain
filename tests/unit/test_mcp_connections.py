import unittest

from src.mcp_clients.insurance_tools import build_mcp_connections


class MCPConnectionsTest(unittest.TestCase):
    def test_retrieval_and_crm_servers_are_registered_together(self):
        connections = build_mcp_connections()
        self.assertEqual(
            set(connections),
            {"insurance_retrieval", "insurance_crm"},
        )
        self.assertIn("retrieval_server.py", connections["insurance_retrieval"]["args"][0])
        self.assertIn("crm_server.py", connections["insurance_crm"]["args"][0])


if __name__ == "__main__":
    unittest.main()
