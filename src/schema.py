"""Canonical document schema shared across every stage of the pipeline.

Keeping one schema means collectors, processing, store, and the dashboard all
speak the same language. A "document" is one collected item (article/post/etc).
"""
from __future__ import annotations

import hashlib
from typing import Any, TypedDict


class Document(TypedDict, total=False):
    id: str            # stable hash of url|title -> used for dedup & vector ids
    source: str        # collector name: "news" | "company" | "hackernews"
    title: str
    text: str
    url: str
    published: str     # ISO-8601 timestamp if known, else ""
    author: str | None
    metadata: dict[str, Any]


def make_doc_id(url: str, title: str) -> str:
    """Deterministic id so re-running collection doesn't create duplicates."""
    key = (url or "").strip().lower() + "|" + (title or "").strip().lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def make_document(
    *,
    source: str,
    title: str,
    text: str,
    url: str = "",
    published: str = "",
    author: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Document:
    """Factory that guarantees a well-formed, id-stamped document."""
    return Document(
        id=make_doc_id(url, title),
        source=source,
        title=title.strip(),
        text=text.strip(),
        url=url,
        published=published,
        author=author,
        metadata=metadata or {},
    )
