"""Tool factory — builds LangChain @tool functions with pre-loaded dependencies
injected via closure so the ReAct agent never reloads config or models per call.

Usage:
    tools = make_tools(cfg, store, embedder)
    llm_with_tools = llm.bind_tools(tools)
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool


def make_tools(cfg: dict[str, Any], store, embedder) -> list:
    from src.agent.rag_chain import RagChain

    rag = RagChain(store, embedder, cfg["intelligence"]["top_k"])

    # ------------------------------------------------------------------
    # Tool 1: semantic search over the ChromaDB knowledge base
    # ------------------------------------------------------------------
    @tool
    def search_kb(query: str) -> str:
        """Search the NVIDIA knowledge base for relevant evidence.
        Use this to retrieve information about NVIDIA's products, strategy,
        financials, partnerships, risks, opportunities, and competitive position.
        Always call this first before drawing any conclusion.
        Input: a focused search query string."""
        chunks = rag.retrieve(query)
        if not chunks:
            return "No relevant chunks found for this query."
        return RagChain.build_context(chunks)

    # ------------------------------------------------------------------
    # Tool 2: fetch fresh news from live Google News RSS
    # ------------------------------------------------------------------
    @tool
    def fetch_live_news(topic: str) -> str:
        """Fetch fresh NVIDIA news articles from live sources.
        Use this when search_kb returns thin or outdated evidence, or when
        you need the most recent developments on a specific topic.
        Input: a topic or keyword (e.g. 'NVIDIA Blackwell GPU launch 2025')."""
        try:
            from src.collectors.news_collector import NewsCollector

            mini_cfg = {
                "company": cfg["company"],
                "collection": {
                    "sources": {
                        "news": {
                            "enabled": True,
                            "type": "google_news",
                            "feeds": cfg["collection"]["sources"]["news"]["feeds"],
                            "queries": [topic],
                            "max_per_query": 8,
                        }
                    }
                },
                "paths": cfg["paths"],
            }
            docs = NewsCollector(mini_cfg).collect()
            if not docs:
                return f"No live news found for '{topic}'."
            lines = [
                f"[{i+1}] {d.get('title', '')} ({(d.get('published') or '')[:10]})\n"
                f"     {(d.get('text') or '')[:200]}"
                for i, d in enumerate(docs[:5])
            ]
            return "\n\n".join(lines)
        except Exception as exc:
            return f"fetch_live_news error: {exc}"

    # ------------------------------------------------------------------
    # Tool 3: non-LLM RoBERTa sentiment over retrieved chunks
    # ------------------------------------------------------------------
    @tool
    def score_sentiment(topic: str) -> str:
        """Retrieve documents related to a topic and score their sentiment
        using a RoBERTa classifier (not the LLM). Use this to gauge market
        or community mood around a specific aspect of NVIDIA's business.
        Input: a topic string (e.g. 'NVIDIA stock outlook', 'GPU supply chain')."""
        try:
            from src.intelligence.sentiment import SentimentAnalyzer

            chunks = rag.retrieve(topic)
            if not chunks:
                return f"No documents found for sentiment scoring on '{topic}'."
            docs = [
                {
                    "id": c.get("chunk_id", ""),
                    "source": c.get("source", ""),
                    "published": c.get("published", ""),
                    "title": c.get("title", ""),
                    "text": c.get("text", ""),
                }
                for c in chunks
            ]
            result = SentimentAnalyzer(cfg).run(docs)
            ov = result.get("overall", {})
            return (
                f"Sentiment for '{topic}' ({ov.get('total', 0)} docs): "
                f"+{ov.get('positive', 0)} positive, "
                f"~{ov.get('neutral', 0)} neutral, "
                f"-{ov.get('negative', 0)} negative "
                f"(net polarity: {ov.get('net_polarity', 0):.3f})"
            )
        except Exception as exc:
            return f"score_sentiment error: {exc}"

    # ------------------------------------------------------------------
    # Tool 4: lightweight metrics from the knowledge base
    # ------------------------------------------------------------------
    @tool
    def compute_metrics(metric_type: str) -> str:
        """Compute a quick metric from the knowledge base.
        Allowed values for metric_type:
          'document_count'   — total chunks indexed
          'source_breakdown' — chunk count per source
          'date_range'       — earliest and latest document dates
        Input: one of the three metric_type strings above."""
        try:
            count = store.count()
            if metric_type == "document_count":
                return f"Total chunks in knowledge base: {count}"

            import numpy as np

            dummy = list(np.zeros(384, dtype=float))
            sample = store.query(dummy, min(count, 200))

            if metric_type == "source_breakdown":
                sources: dict[str, int] = {}
                for c in sample:
                    s = c.get("source", "unknown")
                    sources[s] = sources.get(s, 0) + 1
                return "Source breakdown: " + json.dumps(sources)

            if metric_type == "date_range":
                dates = sorted(
                    c.get("published", "")
                    for c in sample
                    if c.get("published")
                )
                if not dates:
                    return "No date information available in sample."
                return f"Date range: {dates[0][:10]} → {dates[-1][:10]}"

            return (
                f"Unknown metric_type '{metric_type}'. "
                "Choose from: document_count, source_breakdown, date_range."
            )
        except Exception as exc:
            return f"compute_metrics error: {exc}"

    return [search_kb, fetch_live_news, score_sentiment, compute_metrics]
