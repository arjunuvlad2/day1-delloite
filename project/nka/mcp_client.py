"""Sync helpers to talk to the nka MCP server. Work in Jupyter and in plain scripts."""
import asyncio, concurrent.futures, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from .config import ROOT


async def _session(fn):
    params = StdioServerParameters(command=sys.executable, args=["-m", "nka.mcp_server"], cwd=str(ROOT),
                                   env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONWARNINGS": "ignore"})
    with open(os.devnull, "w") as quiet:                       # hide server log noise
        async with stdio_client(params, errlog=quiet) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                return await fn(s)


def with_server(fn, timeout=60):
    """Run `async fn(session)` against a freshly spawned server, in its own thread and event loop."""
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, asyncio.wait_for(_session(fn), timeout)).result()


def list_tools() -> list:
    """[(name, first line of description), ...] for every tool the server exposes."""
    async def go(s): return [(t.name, (t.description or "").split("\n")[0]) for t in (await s.list_tools()).tools]
    return with_server(go)


def call_tool(name: str, **args) -> str:
    """Call one tool on the server and return its text result."""
    async def go(s):
        res = await s.call_tool(name, args)
        return res.content[0].text if res.content else ""
    return with_server(go)


def read_resource(uri: str = "docs://count") -> str:
    async def go(s): return (await s.read_resource(uri)).contents[0].text
    return with_server(go)
