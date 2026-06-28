"""End-to-end pipeline: collect, clean, dedupe, chunk, embed, index, analyse,
recommend, brief, and score sentiment.

Run:  python -m pipeline.run_pipeline
"""
from __future__ import annotations

import os
from typing import Any

from src.agent.briefing import generate_briefing
from src.agent.ceo_agent import CEOAgent
from src.agent.rag_chain import RagChain
from src.collectors.registry import collect_all
from src.config import load_config
from src.intelligence.engine import IntelligenceEngine
from src.llm.client import LLMClient
from src.processing.chunker import chunk_documents
from src.processing.cleaner import clean_documents
from src.processing.deduplicator import dedupe_exact, dedupe_near
from src.report import print_report
from src.schema import Document
from src.store.embeddings import Embedder
from src.store.vector_store import VectorStore
from src.utils import save_json


def run_collection(cfg: dict[str, Any]) -> list[Document]:
    print("== Stage 1: collect ==")
    return collect_all(cfg)


def run_processing(cfg: dict[str, Any], docs: list[Document], embedder: Embedder) -> list[Document]:
    print("== Stage 2: clean + dedupe ==")
    cleaned = clean_documents(docs)
    deduped = dedupe_exact(cleaned)
    print(f"[process] {len(docs)} raw -> {len(cleaned)} cleaned -> {len(deduped)} exact-unique")
    if cfg["processing"].get("near_dedup"):
        lead = [f"{d.get('title','')} {d.get('text','')}"[:600] for d in deduped]
        before = len(deduped)
        deduped = dedupe_near(deduped, embedder.encode(lead), cfg["processing"]["near_dup_threshold"])
        print(f"[process] near-dedup {before} -> {len(deduped)}")
    save_json(deduped, os.path.join(cfg["paths"]["processed"], "clean.json"))
    return deduped


def run_indexing(cfg: dict[str, Any], docs: list[Document], embedder: Embedder) -> None:
    print("== Stage 3-4: chunk + embed + index ==")
    chunks = chunk_documents(docs, cfg)
    save_json(chunks, os.path.join(cfg["paths"]["processed"], "chunks.json"))
    store = VectorStore(cfg)
    store.reset()
    store.add(chunks, embedder.encode([c["text"] for c in chunks]))
    print(f"[index] {len(docs)} docs -> {len(chunks)} chunks -> {store.count()} vectors")


def run_intelligence(cfg: dict[str, Any], embedder: Embedder, llm: LLMClient) -> dict[str, Any]:
    print("== Stage 5: strategic intelligence (LLM) ==")
    rag = RagChain(VectorStore(cfg), embedder, cfg["intelligence"]["top_k"])
    intel = IntelligenceEngine(cfg, rag, llm).run()
    save_json(intel, os.path.join(cfg["paths"]["outputs"], "intelligence.json"))
    print(f"[intel] opportunities={len(intel['opportunities'])} "
          f"risks={len(intel['risks'])} trends={len(intel['trends'])}")
    return intel


def run_strategy(cfg: dict[str, Any], llm: LLMClient, intelligence: dict[str, Any]):
    print("== Stage 6: CEO recommendations + briefing ==")
    recs = CEOAgent(cfg, llm).recommend(intelligence)
    save_json(recs, os.path.join(cfg["paths"]["outputs"], "recommendations.json"))
    brief = generate_briefing(cfg, llm, intelligence, recs)
    save_json(brief, os.path.join(cfg["paths"]["outputs"], "briefing.json"))
    filled = [k for k, v in brief.items() if v]
    print(f"[strategy] {len(recs)} recommendations; briefing sections filled: {filled}")
    return recs, brief


def run_sentiment(cfg: dict[str, Any], docs: list[Document]) -> dict[str, Any]:
    print("== Stage 7: sentiment analysis ==")
    from src.intelligence.sentiment import SentimentAnalyzer

    result = SentimentAnalyzer(cfg).run(docs)
    save_json(result, os.path.join(cfg["paths"]["outputs"], "sentiment.json"))
    ov = result["overall"]
    print(f"[sentiment] +{ov['positive']} ~{ov['neutral']} -{ov['negative']} "
          f"(net {ov['net_polarity']})")
    return result


def main() -> None:
    cfg = load_config()
    embedder = Embedder(cfg["store"]["embedding_model"])
    llm = LLMClient(cfg)

    docs = run_collection(cfg)
    processed = run_processing(cfg, docs, embedder)
    run_indexing(cfg, processed, embedder)
    intel = run_intelligence(cfg, embedder, llm)
    recs, brief = run_strategy(cfg, llm, intel)
    sentiment = run_sentiment(cfg, processed)

    print_report(cfg, intel, recs, brief, sentiment)
    print("[done] dashboard:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
