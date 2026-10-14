"""Evaluation helpers: retrieval Recall@k by query type, refusal check, agent runs."""
import json
import pandas as pd
from .config import EVAL_FILE, MIN_SIM
from .data import load_docs, chunk_docs, superseded_doc_ids
from .index import Index


def golden_queries() -> list:
    """All queries from data/eval_queries.json (types: semantic, keyword, hybrid, no_answer)."""
    return json.load(open(EVAL_FILE, encoding="utf-8"))


def evaluate_retrieval(chunk_words: int = 150, k: int = 3, role: str = "hr_partner", fresh_only: bool = False) -> dict:
    """Recall@k by query type (a hit = any relevant doc among the top-k chunks) plus refusal accuracy.

    Superseded documents are kept by default because the gold set marks them relevant.
    """
    docs = load_docs()
    idx = Index(superseded_doc_ids(docs)).build(chunk_docs(docs, chunk_words))
    hits, refusals = {}, []
    for q in golden_queries():
        res = idx.search(q["query"], k=k, role=role, fresh_only=fresh_only)
        if q["type"] == "no_answer":
            refusals.append(bool(res) and res[0]["score"] < MIN_SIM)
        else:
            hits.setdefault(q["type"], []).append(any(r["doc_id"] in q["relevant_doc_ids"] for r in res))
    per_type = {t: round(sum(h) / len(h), 3) for t, h in hits.items()}
    allh = sum(hits.values(), [])
    return dict(recall_at_k=per_type, overall=round(sum(allh) / len(allh), 3),
                refusal_correct=f"{sum(refusals)}/{len(refusals)}", chunks=len(idx.chunks))


def evaluate_agent(questions: list, **kw) -> pd.DataFrame:
    """Run the agent on each question quietly; one row each: question, answer, steps, tool_calls."""
    from .agent import run_agent
    rows = []
    for q in questions:
        r = run_agent(q, verbose=False, **kw)
        rows.append(dict(question=q, answer=r["answer"], steps=" > ".join(s["action"] for s in r["steps"]), tool_calls=r["tool_calls"]))
    return pd.DataFrame(rows)
