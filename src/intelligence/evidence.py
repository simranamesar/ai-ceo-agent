"""Turn retrieved chunks into citable evidence records."""
from __future__ import annotations

from typing import Any


def build_evidence(retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """De-duplicate by parent document and trim to a citable snippet."""
    seen: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for c in retrieved_chunks:
        key = c.get("doc_id") or c.get("chunk_id") or c.get("url", "")
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            {
                "source": c.get("source", ""),
                "title": c.get("title", ""),
                "url": c.get("url", ""),
                "snippet": (c.get("text", "") or "")[:300],
                "published": c.get("published", ""),
            }
        )
    return evidence
