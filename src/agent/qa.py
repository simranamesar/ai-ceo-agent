"""Interactive 'Ask the CEO' Q&A (dashboard).

Answers a free-form strategic question by retrieving the most relevant chunks
from the knowledge base and having the LLM reason over them, so the answer is
grounded in cited evidence rather than the model's prior knowledge.
"""
from __future__ import annotations

from typing import Any

from src.agent import prompts
from src.agent.rag_chain import RagChain
from src.intelligence.evidence import build_evidence
from src.llm.client import LLMClient
from src.store.embeddings import Embedder
from src.store.vector_store import VectorStore


def ask_ceo(
    cfg: dict[str, Any],
    question: str,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Return {'answer': str, 'evidence': [...]} for a strategic question."""
    embedder = embedder or Embedder(cfg["store"]["embedding_model"])
    store = store or VectorStore(cfg)
    llm = llm or LLMClient(cfg)

    rag = RagChain(store, embedder, cfg["intelligence"]["top_k"])
    chunks = rag.retrieve(question)
    context = rag.build_context(chunks)

    system = prompts.ASK_SYSTEM.format(company=cfg["company"]["name"])
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    answer = llm.chat(system, user)
    return {"answer": (answer or "").strip(), "evidence": build_evidence(chunks)}
