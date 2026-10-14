"""Load the corpus, split it into chunks, find superseded versions."""
import re
from .config import CORPUS_DIR


def load_docs() -> list:
    """One dict per corpus file: doc_id, title, department, date, sensitivity, policy_id, body."""
    docs = []
    for f in sorted(CORPUS_DIR.glob("*.txt")):
        head, body = f.read_text(encoding="utf-8").split("\n\n", 1)
        meta = dict(l.split(": ", 1) for l in head.splitlines() if ": " in l)
        docs.append(dict(doc_id=f.name.split("_")[0], title=meta.get("title", f.stem),
                         department=meta.get("department", ""), date=meta.get("date", ""),
                         sensitivity=meta.get("sensitivity", "internal"),
                         policy_id=meta.get("policy_id", ""), body=body.strip()))
    return docs


def chunk_docs(docs: list, max_words: int = 150) -> list:
    """Split each body into pieces of at most max_words words; max_words=0 keeps whole documents."""
    chunks = []
    for d in docs:
        words = re.findall(r"\S+\s*", d["body"])
        step = max_words if max_words else max(len(words), 1)
        for n, i in enumerate(range(0, len(words), step)):
            chunks.append(dict(chunk_id=f"{d['doc_id']}-{n}", doc_id=d["doc_id"], title=d["title"],
                               text="".join(words[i:i + step]).strip(), sensitivity=d["sensitivity"],
                               date=d["date"], policy_id=d["policy_id"]))
    return chunks


def superseded_doc_ids(docs: list) -> list:
    """Doc ids that are older versions of a policy_id that has a newer document."""
    latest = {}
    for d in docs:
        if d["policy_id"]: latest[d["policy_id"]] = max(latest.get(d["policy_id"], ""), d["date"])
    return sorted(d["doc_id"] for d in docs if d["policy_id"] and d["date"] < latest[d["policy_id"]])
