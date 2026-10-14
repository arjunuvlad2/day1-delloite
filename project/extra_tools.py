"""Registers lookup_policy in the MCP server process (NKA_EXTRA_TOOLS=extra_tools)."""
from nka.tools import register_tool, lookup_policy

register_tool("lookup_policy", lookup_policy, "find a policy by its ID, e.g. POL-HR-017; input = the policy ID")
