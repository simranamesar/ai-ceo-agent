"""Graph 1 — the offline pipeline as a LangGraph StateGraph.

Nodes run in a fixed linear order (collect → process → index → intelligence
→ sentiment → recommend → verify → brief). Making this a LangGraph makes the
execution plan explicit and visible: you can print the graph, inspect which node
ran, and extend it with conditional edges later.

Run:
    python -m pipeline.graph
"""
from __future__ import annotations

import os
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agent.briefing import generate_briefing
from src.agent.ceo_agent import CEOAgent
from src.agent.rag_chain import RagChain
from src.agent.verifier import verify_recommendations
from src.collectors.registry import collect_all
from src.config import load_config
from src.intelligence.engine import IntelligenceEngine
from src.intelligence.sentiment import SentimentAnalyzer
from src.llm.client import LLMClient
from src.processing.chunker import chunk_documents
from src.processing.cleaner import clean_documents
from src.processing.deduplicator import dedupe_exact, dedupe_near
from src.report import print_report
from src.store.embeddings import Embedder
from src.store.vector_store import VectorStore
from src.utils import save_json


# ---------------------------------------------------------------------------
# Shared state that flows through every node
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    cfg: dict[str, Any]
    embedder: Any          # Embedder — kept as Python object, not serialised
    llm: Any               # LLMClient — same
    docs: list[dict]       # raw collected documents
    processed: list[dict]  # cleaned + deduped documents
    intelligence: dict[str, Any]
    sentiment: dict[str, Any]
    recommendations: list[dict]
    verification: dict[str, Any]
    briefing: dict[str, Any]


# ---------------------------------------------------------------------------
# Nodes — each receives the full state, returns only what it changed
# ---------------------------------------------------------------------------

def collect_node(state: PipelineState) -> dict:
    print("== [Graph1] collect ==")
    docs = collect_all(state["cfg"])
    raw_path = os.path.join(state["cfg"]["paths"]["raw"], "documents.json")
    save_json(docs, raw_path)
    print(f"[collect] {len(docs)} documents")
    return {"docs": docs}


def process_node(state: PipelineState) -> dict:
    print("== [Graph1] process ==")
    cfg, embedder = state["cfg"], state["embedder"]
    cleaned = clean_documents(state["docs"])
    deduped = dedupe_exact(cleaned)
    print(f"[process] {len(state['docs'])} raw → {len(cleaned)} cleaned → {len(deduped)} exact-unique")
    if cfg["processing"].get("near_dedup"):
        lead = [f"{d.get('title','')} {d.get('text','')}"[:600] for d in deduped]
        before = len(deduped)
        deduped = dedupe_near(deduped, embedder.encode(lead), cfg["processing"]["near_dup_threshold"])
        print(f"[process] near-dedup {before} → {len(deduped)}")
    save_json(deduped, os.path.join(cfg["paths"]["processed"], "clean.json"))
    return {"processed": deduped}


def index_node(state: PipelineState) -> dict:
    print("== [Graph1] index ==")
    cfg, embedder = state["cfg"], state["embedder"]
    chunks = chunk_documents(state["processed"], cfg)
    save_json(chunks, os.path.join(cfg["paths"]["processed"], "chunks.json"))
    store = VectorStore(cfg)
    store.reset()
    store.add(chunks, embedder.encode([c["text"] for c in chunks]))
    print(f"[index] {len(chunks)} chunks → {store.count()} vectors")
    return {}  # side effect: VectorStore written to disk


def intelligence_node(state: PipelineState) -> dict:
    print("== [Graph1] intelligence ==")
    cfg, embedder, llm = state["cfg"], state["embedder"], state["llm"]
    rag = RagChain(VectorStore(cfg), embedder, cfg["intelligence"]["top_k"])
    intel = IntelligenceEngine(cfg, rag, llm).run()
    save_json(intel, os.path.join(cfg["paths"]["outputs"], "intelligence.json"))
    print(
        f"[intel] opportunities={len(intel.get('opportunities', []))} "
        f"risks={len(intel.get('risks', []))} "
        f"trends={len(intel.get('trends', []))}"
    )
    return {"intelligence": intel}


def sentiment_node(state: PipelineState) -> dict:
    print("== [Graph1] sentiment ==")
    result = SentimentAnalyzer(state["cfg"]).run(state["processed"])
    save_json(result, os.path.join(state["cfg"]["paths"]["outputs"], "sentiment.json"))
    ov = result.get("overall", {})
    print(f"[sentiment] +{ov.get('positive',0)} ~{ov.get('neutral',0)} -{ov.get('negative',0)} "
          f"(net {ov.get('net_polarity', 0)})")
    return {"sentiment": result}


def recommend_node(state: PipelineState) -> dict:
    print("== [Graph1] recommend ==")
    recs = CEOAgent(state["cfg"], state["llm"]).recommend(state["intelligence"])
    print(f"[recommend] {len(recs)} recommendations drafted")
    return {"recommendations": recs}


def verify_node(state: PipelineState) -> dict:
    """Validation gate: drop recommendations with < 3 evidence pieces."""
    print("== [Graph1] verify ==")
    report = verify_recommendations(state["recommendations"])
    # Write only the verified set to the dashboard artifact
    save_json(report["verified"], os.path.join(state["cfg"]["paths"]["outputs"], "recommendations.json"))
    save_json(report, os.path.join(state["cfg"]["paths"]["outputs"], "verification.json"))
    print(
        f"[verify] {report['passed']}/{report['total']} passed "
        f"({report['failed']} rejected for insufficient evidence)"
    )
    return {"recommendations": report["verified"], "verification": report}


def brief_node(state: PipelineState) -> dict:
    print("== [Graph1] brief ==")
    brief = generate_briefing(
        state["cfg"], state["llm"], state["intelligence"], state["recommendations"]
    )
    save_json(brief, os.path.join(state["cfg"]["paths"]["outputs"], "briefing.json"))
    return {"briefing": brief}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_pipeline():
    """Compile and return the Graph 1 LangGraph pipeline."""
    g = StateGraph(PipelineState)

    nodes = [
        ("collect", collect_node),
        ("process", process_node),
        ("index", index_node),
        ("intelligence", intelligence_node),
        ("sentiment", sentiment_node),
        ("recommend", recommend_node),
        ("verify", verify_node),
        ("brief", brief_node),
    ]
    for name, fn in nodes:
        g.add_node(name, fn)

    g.set_entry_point("collect")
    for (a, _), (b, _) in zip(nodes, nodes[1:]):
        g.add_edge(a, b)
    g.add_edge("brief", END)

    return g.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    embedder = Embedder(cfg["store"]["embedding_model"])
    llm = LLMClient(cfg)

    pipeline = build_pipeline()
    final = pipeline.invoke(
        {
            "cfg": cfg,
            "embedder": embedder,
            "llm": llm,
            "docs": [],
            "processed": [],
            "intelligence": {},
            "sentiment": {},
            "recommendations": [],
            "verification": {},
            "briefing": {},
        }
    )

    print_report(
        cfg,
        final["intelligence"],
        final["recommendations"],
        final["briefing"],
        final["sentiment"],
    )
    print("[done] dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
