"""Embedding index with role-based security trimming and a freshness filter."""
import hashlib
import warnings
import numpy as np
from .config import ARTIFACTS, MODEL_NAME, ROLES
from .data import load_docs, chunk_docs, superseded_doc_ids

warnings.filterwarnings("ignore")
_model = None
_CACHE_FILE = ARTIFACTS / "embedding_cache.npz"
_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        try:
            z = np.load(_CACHE_FILE)
            _cache = dict(zip(z["keys"].tolist(), z["vecs"]))
        except Exception:
            _cache = {}
    return _cache


def _save_cache():
    if _cache:
        np.savez(_CACHE_FILE, keys=np.array(list(_cache)), vecs=np.array(list(_cache.values())))


def _key(t): return hashlib.sha1((MODEL_NAME + "|" + t).encode()).hexdigest()


def embed(texts) -> np.ndarray:
    """Normalised embeddings, shape (n, 384). Cosine similarity = dot product. Cached on disk by text hash."""
    global _model
    texts, cache = list(texts), _load_cache()
    missing = [t for t in dict.fromkeys(texts) if _key(t) not in cache]
    if missing:
        if _model is None:
            from fastembed import TextEmbedding
            _model = TextEmbedding(MODEL_NAME)
        for t, v in zip(missing, _model.embed(missing)):
            v = np.asarray(v, dtype=float)
            cache[_key(t)] = v / np.linalg.norm(v)
        _save_cache()
    return np.array([cache[_key(t)] for t in texts])


class Index:
    """Vector index over chunks. search() trims by role BEFORE ranking."""

    def __init__(self, superseded=()):
        self.chunks, self.vecs, self.superseded = [], np.zeros((0, 384)), set(superseded)

    def embed(self, texts): return embed(texts)

    def build(self, chunks: list):
        self.chunks = list(chunks)
        self.vecs = self.embed([c["text"] for c in self.chunks])
        return self

    def search(self, query: str, k: int = 3, role: str = "employee", fresh_only: bool = True) -> list:
        if role not in ROLES: raise ValueError(f"unknown role '{role}': access denied")
        allowed = [i for i, c in enumerate(self.chunks) if c["sensitivity"] in ROLES[role]
                   and not (fresh_only and c["doc_id"] in self.superseded)]
        if not allowed: return []
        sims = self.vecs[allowed] @ self.embed([query])[0]
        top = np.argsort(-sims)[:k]
        return [dict(self.chunks[allowed[j]], score=float(sims[j])) for j in top]


_INDEX = None


def get_index() -> Index:
    """Build the default index (150-word chunks of the corpus) once per process."""
    global _INDEX
    if _INDEX is None:
        docs = load_docs()
        _INDEX = Index(superseded_doc_ids(docs)).build(chunk_docs(docs, 150))
    return _INDEX
