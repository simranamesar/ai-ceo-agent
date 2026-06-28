"""Strategic intelligence engine: RAG retrieval + the LLM surface opportunities,
risks and trends. Each item is mapped back to the source chunks it cited, so
findings carry traceable evidence and a retrieval-derived confidence score.
"""
from __future__ import annotations

from typing import Any

from src.agent import prompts
from src.agent.rag_chain import RagChain
from src.intelligence.evidence import build_evidence
from src.llm.client import LLMClient
from src.utils import now_iso

_QUERIES = {
    "opportunities": "growth opportunities, new markets, partnerships, "
    "emerging technologies, product opportunities",
    "risks": "competitive threats, regulatory changes, supply chain issues, "
    "negative sentiment, financial risk",
    "trends": "technology trends, customer behaviour shifts, industry developments",
}


class IntelligenceEngine:
    def __init__(self, cfg: dict[str, Any], rag: RagChain, llm: LLMClient) -> None:
        self.cfg = cfg
        self.rag = rag
        self.llm = llm
        self.company = cfg["company"]["name"]

    # -- one category extraction ----------------------------------------
    def _extract(self, kind: str, system_template: str) -> list[dict[str, Any]]:
        query = f"{self.company} {_QUERIES[kind]}"
        chunks = self.rag.retrieve(query)
        context = self.rag.build_context(chunks)
        system = system_template.format(company=self.company)
        user = f"CONTEXT:\n{context}\n\nReturn ONLY the JSON array described."
        try:
            items = self.llm.json_chat(system, user)
        except Exception as e:
            print(f"[intel] {kind}: extraction/parse failed: {e}")
            return []
        if isinstance(items, dict):
            items = items.get("items", [])
        if not isinstance(items, list):
            return []
        return [self._finalize(it, chunks) for it in items if isinstance(it, dict)]

    # -- attach real evidence + confidence ------------------------------
    def _finalize(self, item: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
        ev_chunks = []
        for n in item.get("evidence", []) or []:
            try:
                k = int(n) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= k < len(chunks):
                ev_chunks.append(chunks[k])
        if not ev_chunks:
            ev_chunks = chunks[:3]
        item["confidence"] = self._confidence(item, ev_chunks)
        item["evidence"] = build_evidence(ev_chunks)
        return item

    @staticmethod
    def _confidence(item: dict[str, Any], ev_chunks: list[dict[str, Any]]) -> float:
        c = item.get("confidence")
        if isinstance(c, (int, float)) and 0 < c <= 1:   # treat 0.0 / missing as unset
            return round(float(c), 2)
        dists = [ch["distance"] for ch in ev_chunks if isinstance(ch.get("distance"), (int, float))]
        if dists:
            sim = 1 - sum(dists) / len(dists)            # cosine distance -> similarity
            return round(max(0.3, min(1.0, sim)), 2)     # floored so a cited finding is never 0
        return 0.6

    # -- public API ------------------------------------------------------
    def extract_opportunities(self) -> list[dict[str, Any]]:
        return self._extract("opportunities", prompts.OPPORTUNITY_SYSTEM)

    def extract_risks(self) -> list[dict[str, Any]]:
        return self._extract("risks", prompts.RISK_SYSTEM)

    def extract_trends(self) -> list[dict[str, Any]]:
        return self._extract("trends", prompts.TREND_SYSTEM)

    def run(self) -> dict[str, Any]:
        return {
            "company": self.company,
            "generated_at": now_iso(),
            "opportunities": self.extract_opportunities(),
            "risks": self.extract_risks(),
            "trends": self.extract_trends(),
        }
