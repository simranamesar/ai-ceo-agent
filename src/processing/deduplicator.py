"""Remove exact duplicates (by id) and near-duplicates (by cosine similarity)."""
from __future__ import annotations

from src.schema import Document


def dedupe_exact(docs: list[Document]) -> list[Document]:
    """Drop docs sharing the same id (same url|title hash)."""
    seen: set[str] = set()
    out: list[Document] = []
    for d in docs:
        if d["id"] in seen:
            continue
        seen.add(d["id"])
        out.append(d)
    return out


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def dedupe_near(
    docs: list[Document], embeddings: list[list[float]], threshold: float
) -> list[Document]:
    """Greedy near-duplicate removal. Assumes L2-normalised embeddings so the
    dot product equals cosine similarity. Keeps a doc unless it is >= threshold
    similar to an already-kept doc."""
    kept: list[Document] = []
    kept_vecs: list[list[float]] = []
    for d, v in zip(docs, embeddings):
        if any(_dot(v, kv) >= threshold for kv in kept_vecs):
            continue
        kept.append(d)
        kept_vecs.append(v)
    return kept
