"""Document-level sentiment with a transformer encoder classifier (RoBERTa),
aggregated into overall, news-vs-public, and a time trend. aggregate() is a
pure function (no model), so it can be unit-tested offline.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

from src.schema import Document

_LABEL_MAP = {
    "label_0": "negative", "label_1": "neutral", "label_2": "positive",
    "pos": "positive", "neg": "negative", "neu": "neutral",
}


def _norm_label(label: str) -> str:
    l = (label or "").strip().lower()
    if l in _LABEL_MAP:
        return _LABEL_MAP[l]
    if "pos" in l:
        return "positive"
    if "neg" in l:
        return "negative"
    return "neutral"


def _parse_date(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        pass
    try:
        return parsedate_to_datetime(s).date()
    except (ValueError, TypeError):
        return None


class SentimentAnalyzer:
    def __init__(self, cfg: dict[str, Any]) -> None:
        s = cfg.get("sentiment", {}) or {}
        self.model_name = s.get("model", "cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.device = s.get("device", "cpu")
        self.batch_size = s.get("batch_size", 16)
        self.news_sources = set(s.get("news_sources", ["news", "company"]))
        self.public_sources = set(s.get("public_sources", ["hackernews"]))
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            from transformers import pipeline

            device = -1 if self.device == "cpu" else self.device
            self._pipe = pipeline(
                "sentiment-analysis", model=self.model_name, device=device,
                truncation=True, max_length=512,
            )
        return self._pipe

    def score_documents(self, docs: list[Document]) -> list[dict[str, Any]]:
        if not docs:
            return []
        pipe = self._ensure()
        texts = [f"{d.get('title','')}. {d.get('text','') or ''}"[:1000] for d in docs]
        results = pipe(texts, batch_size=self.batch_size)
        scored = []
        for d, r in zip(docs, results):
            scored.append(
                {
                    "doc_id": d.get("id", ""),
                    "source": d.get("source", ""),
                    "published": d.get("published", ""),
                    "label": _norm_label(r.get("label", "")),
                    "score": float(r.get("score", 0.0)),
                }
            )
        return scored

    @staticmethod
    def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        for r in rows:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        total = sum(counts.values())
        net = round((counts["positive"] - counts["negative"]) / total, 3) if total else 0.0
        return {**counts, "total": total, "net_polarity": net}

    def aggregate(self, scored: list[dict[str, Any]]) -> dict[str, Any]:
        buckets: dict[str, list] = defaultdict(list)
        for r in scored:
            d = _parse_date(r.get("published", ""))
            if d:
                buckets[d.isoformat()].append(r)
        trend = [
            {"date": day, "net_polarity": self._agg(buckets[day])["net_polarity"], "count": len(buckets[day])}
            for day in sorted(buckets)
        ]
        by_source = {
            src: self._agg([r for r in scored if r.get("source") == src])
            for src in sorted({r.get("source", "") for r in scored if r.get("source")})
        }
        return {
            "overall": self._agg(scored),
            "by_category": {
                "news": self._agg([r for r in scored if r["source"] in self.news_sources]),
                "public": self._agg([r for r in scored if r["source"] in self.public_sources]),
            },
            "by_source": by_source,
            "trend": trend,
        }

    def run(self, docs: list[Document]) -> dict[str, Any]:
        return self.aggregate(self.score_documents(docs))
