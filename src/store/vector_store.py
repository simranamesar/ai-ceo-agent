"""Persistent ChromaDB vector store: metadata + a cosine index. Embeddings are
supplied explicitly, so Chroma never downloads its own embedding model.
"""
from __future__ import annotations

from typing import Any


class VectorStore:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.persist_dir = cfg["store"]["persist_dir"]
        self.collection_name = cfg["store"]["collection_name"]
        self._client = None
        self._collection = None

    def _coll(self):
        if self._collection is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # build a cosine index
            )
        return self._collection

    def add(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        coll = self._coll()
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "doc_id": c.get("doc_id", "") or "",
                "source": c.get("source", "") or "",
                "title": c.get("title", "") or "",
                "url": c.get("url", "") or "",
                "published": c.get("published", "") or "",
            }
            for c in chunks
        ]
        batch = 1000
        for i in range(0, len(ids), batch):
            coll.upsert(
                ids=ids[i : i + batch],
                documents=documents[i : i + batch],
                embeddings=embeddings[i : i + batch],
                metadatas=metadatas[i : i + batch],
            )

    def query(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        coll = self._coll()
        res = coll.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out: list[dict[str, Any]] = []
        ids = res["ids"][0]
        for i in range(len(ids)):
            meta = res["metadatas"][0][i] or {}
            out.append(
                {
                    "chunk_id": ids[i],
                    "text": res["documents"][0][i],
                    "distance": res["distances"][0][i],
                    **meta,
                }
            )
        return out

    def count(self) -> int:
        return self._coll().count()

    def reset(self) -> None:
        """Drop and recreate the collection so each pipeline run is clean."""
        import chromadb

        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None
