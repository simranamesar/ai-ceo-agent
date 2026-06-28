"""Split documents into overlapping char windows for embedding/retrieval.

Each chunk keeps a back-reference to its parent document (doc_id + source +
url) so retrieved chunks can be turned into citable evidence later.
chunk_size / chunk_overlap / min_chunk_chars come from config['processing'].
"""
from __future__ import annotations

from typing import Any

from src.schema import Document


def _windows(text: str, size: int, overlap: int) -> list[str]:
    """Deterministic sliding window. Short text -> a single whole window."""
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    out: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        out.append(text[start:end])
        if end == n:
            break
        start += step
    return out


def chunk_documents(docs: list[Document], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    p = cfg["processing"]
    size, overlap, min_chars = p["chunk_size"], p["chunk_overlap"], p["min_chunk_chars"]
    chunks: list[dict[str, Any]] = []
    for d in docs:
        text = (d.get("text") or "").strip()
        if not text:
            continue
        wins = _windows(text, size, overlap)
        # Drop tiny trailing fragments, but never drop a doc's only window.
        if len(wins) > 1:
            wins = [w for w in wins if len(w) >= min_chars] or wins[:1]
        for i, w in enumerate(wins):
            chunks.append(
                {
                    "chunk_id": f"{d['id']}_{i}",
                    "doc_id": d["id"],
                    "source": d.get("source", ""),
                    "title": d.get("title", ""),
                    "url": d.get("url", ""),
                    "published": d.get("published", ""),
                    "text": w.strip(),
                }
            )
    return chunks
