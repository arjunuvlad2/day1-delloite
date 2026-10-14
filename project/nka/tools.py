"""The agent's tools and the registry that the agent and the MCP server both read."""
import ast, json, operator, re, sqlite3
from .data import load_docs
from .index import get_index


def search_docs(query: str, k: int = 3, role: str = "employee") -> str:
    """Search the company documents; returns the top-k chunks with source title and similarity."""
    hits = get_index().search(query, k=k, role=role)
    return "\n".join(f"[{h['title']}] (sim {h['score']:.2f}) {h['text']}" for h in hits) or "No documents available for this role."


# In-memory bookings table (the London hotel cap in the policy corpus is GBP 210 per night)
_DB = sqlite3.connect(":memory:", check_same_thread=False)
_DB.execute("CREATE TABLE bookings(city TEXT, traveller TEXT, nights INTEGER, price_per_night REAL, currency TEXT)")
_DB.executemany("INSERT INTO bookings VALUES (?,?,?,?,?)", [
    ("London", "A. Shah", 3, 195, "GBP"), ("London", "B. Okafor", 2, 240, "GBP"),
    ("London", "C. Meyer", 4, 210, "GBP"), ("London", "D. Rossi", 3, 265, "GBP"),
    ("London", "E. Tan", 1, 180, "GBP"), ("London", "F. Novak", 2, 225, "GBP"),
    ("Paris", "G. Lopez", 3, 250, "EUR"), ("Paris", "H. Kim", 2, 230, "EUR"),
    ("New York", "I. Brown", 4, 360, "USD"), ("New York", "J. Singh", 2, 320, "USD"),
    ("Manchester", "K. Evans", 2, 140, "GBP"), ("Dublin", "L. Byrne", 3, 215, "EUR")])
SCHEMA = "bookings(city TEXT, traveller TEXT, nights INTEGER, price_per_night REAL, currency TEXT)"


def query_sql(sql: str, limit: int = 20) -> str:
    """Read-only SQL on the bookings table: one SELECT statement, at most `limit` rows, returned as JSON."""
    s = sql.strip().rstrip(";")
    if ";" in s or not re.match(r"(?i)^select\b", s): raise ValueError("only a single SELECT statement is allowed")
    if re.search(r"(?i)\b(insert|update|delete|drop|alter|create|attach|pragma|replace)\b", s):
        raise ValueError("write keywords are blocked")
    cur = _DB.execute(s)
    cols = [c[0] for c in cur.description]
    return json.dumps([dict(zip(cols, r)) for r in cur.fetchmany(limit)])


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod}


def calculator(expression: str) -> str:
    """Safe arithmetic only: numbers and + - * / ** % (parsed with ast, never eval)."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("only plain arithmetic is allowed")
    return str(round(ev(ast.parse(expression, mode="eval").body), 4))


def lookup_policy(policy_id: str) -> str:
    """Find a policy by its ID (e.g. POL-HR-017); returns the title and the first two sentences."""
    pid = policy_id.strip().upper()
    found = [d for d in load_docs() if d["policy_id"].upper() == pid]
    if not found:      # fall back to the ID appearing in the body text
        found = [d for d in load_docs() if re.search(rf"\b{re.escape(pid)}\b", d["body"], re.I)]
    if not found: return f"Policy {policy_id} not found."
    d = max(found, key=lambda x: x["date"])            # newest version
    body = " ".join(l for l in d["body"].splitlines() if l.strip() and not l.lower().startswith("policy id"))
    sents = re.split(r"(?<=[.!?])\s+", re.sub(r"\b\d+\.\s+(?=[A-Z])", "", body))[:2]
    return f"{d['title']} ({d['policy_id'] or pid}, {d['date']}): " + " ".join(sents)


TOOLS = {
    "search_docs": dict(fn=search_docs, description="search the company documents; input = a search query"),
    "query_sql": dict(fn=query_sql, description=f"read-only SQL (one SELECT) on {SCHEMA}; input = a SELECT statement"),
    "calculator": dict(fn=calculator, description="evaluate arithmetic such as 210 * 3; input = an expression"),
}


def register_tool(name: str, fn, description: str):
    """Add a tool. The agent can then use it and the MCP server exposes it automatically."""
    TOOLS[name] = dict(fn=fn, description=description)
