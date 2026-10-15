"""Sync helpers to talk to the nka MCP server. Work in Jupyter and in plain scripts."""
import asyncio, concurrent.futures, os, subprocess, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from .config import ROOT, ARTIFACTS

LOG = ARTIFACTS / "mcp_server.log"
_checked = False


def _env():
    return {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONWARNINGS": "ignore"}


def _preflight():
    """Start the server module once in a plain subprocess so any start-up error is shown clearly (not buried in a task group)."""
    global _checked
    if _checked and not os.environ.get("NKA_EXTRA_TOOLS"): return
    try:
        r = subprocess.run([sys.executable, "-c", "import nka.mcp_server"], cwd=str(ROOT), env=_env(),
                           capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        raise RuntimeError("The MCP server took more than 3 minutes to import (is the embedding model still downloading?). Run M1 once first, then retry.") from None
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-12:])
        raise RuntimeError("The MCP server could not start. Last lines of its error:\n" + tail +
                           "\n\nCheck: you started Jupyter from the project folder, and `pip install -r requirements.txt` finished.") from None
    _checked = True


def _leaves(e):
    subs = getattr(e, "exceptions", None)
    return [x for s in subs for x in _leaves(s)] if subs else [e]


async def _session(fn):
    params = StdioServerParameters(command=sys.executable, args=["-m", "nka.mcp_server"], cwd=str(ROOT), env=_env())
    with open(LOG, "w") as log:                                  # server messages go to artifacts/mcp_server.log
        async with stdio_client(params, errlog=log) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                return await fn(s)


def with_server(fn, timeout=60):
    """Run `async fn(session)` against a freshly spawned server, in its own thread and event loop."""
    _preflight()
    try:
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            return ex.submit(asyncio.run, asyncio.wait_for(_session(fn), timeout)).result()
    except BaseException as e:
        leaves = _leaves(e)
        msg = "; ".join(f"{type(x).__name__}: {x}" for x in leaves)[:600]
        try: tail = "\n".join(LOG.read_text().strip().splitlines()[-10:])
        except Exception: tail = ""
        raise RuntimeError(f"MCP call failed: {msg}" + (f"\nServer log (artifacts/mcp_server.log):\n{tail}" if tail else "")) from None


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
