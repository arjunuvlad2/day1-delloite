"""nka: the Northwind Knowledge Assistant building blocks.

Nothing heavy is imported here. Use the submodules directly, e.g.
    from nka.index import get_index
    from nka.agent import run_agent
"""
import importlib

__all__ = ["config", "data", "index", "tools", "agent", "security", "evalkit", "monitor", "risk",
           "mcp_server", "mcp_client", "llm_client"]


def __getattr__(name):          # lazy: `import nka; nka.agent` loads the module on first use
    if name in __all__:
        return importlib.import_module(f"nka.{name}")
    raise AttributeError(name)
