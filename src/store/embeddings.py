"""Embedding model wrapper (sentence-transformers).

Model from config['store']['embedding_model'] (default all-MiniLM-L6-v2).
Embeddings are L2-normalised so cosine == dot product (used by near-dedup and
the cosine-space Chroma index). Lazy-loads so importing this module is cheap.
"""
from __future__ import annotations


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure()
        vecs = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    @property
    def dim(self) -> int:
        return len(self.encode(["dimension probe"])[0])
