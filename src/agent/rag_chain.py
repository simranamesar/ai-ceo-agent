"""RAG retrieval chain: embed a query, pull top-k chunks, format a numbered,
citable context block. Shared by the Intelligence Engine and the CEO agent.
"""
from __future__ import annotations

from typing import Any

from src.store.embeddings import Embedder
from src.store.vector_store import VectorStore


class RagChain:
    def __init__(self, store: VectorStore, embedder: Embedder, top_k: int) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        qvec = self.embedder.encode([query])[0]
        return self.store.query(qvec, self.top_k)

    @staticmethod
    def build_context(chunks: list[dict[str, Any]]) -> str:
        blocks = []
        for i, c in enumerate(chunks, 1):
            blocks.append(
                f"[{i}] ({c.get('source','')}) {c.get('title','')}\n"
                f"{c.get('text','')}\n"
                f"URL: {c.get('url','')}"
            )
        return "\n\n".join(blocks)
