"""MCP server 'nka': exposes every tool in nka.tools.TOOLS plus a docs://count resource.

Run over stdio:  python -m nka.mcp_server
Extra tools: set NKA_EXTRA_TOOLS=some_module; importing it may call nka.tools.register_tool(...).
"""
import importlib, os
from mcp.server.fastmcp import FastMCP
from .data import load_docs
from .tools import TOOLS


def build_server() -> FastMCP:
    """Create the server from the CURRENT registry (so registered tools are exposed automatically)."""
    if os.environ.get("NKA_EXTRA_TOOLS"):
        importlib.import_module(os.environ["NKA_EXTRA_TOOLS"])
    server = FastMCP("nka")
    for name, t in TOOLS.items():
        server.tool(name=name, description=t["description"])(t["fn"])

    @server.resource("docs://count")
    def doc_count() -> str:
        """How many documents the assistant can search."""
        return str(len(load_docs()))

    return server


mcp = build_server()

if __name__ == "__main__":
    mcp.run(transport="stdio")
