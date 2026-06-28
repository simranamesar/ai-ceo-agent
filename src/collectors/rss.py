"""Map a feedparser entry (or a dict shaped like one) -> Document.

Shared by every RSS-based collector so parsing lives in one place and can be
unit-tested without hitting the network.
"""
from __future__ import annotations

from typing import Any

from src.schema import Document, make_document


def rss_entry_to_doc(entry: Any, source: str) -> Document:
    title = entry.get("title", "") or ""
    link = entry.get("link", "") or ""
    published = entry.get("published", "") or entry.get("updated", "") or ""

    content = ""
    if entry.get("content"):
        try:
            content = entry["content"][0].get("value", "")
        except (AttributeError, IndexError, KeyError, TypeError):
            content = ""
    text = content or entry.get("summary", "") or title

    src = entry.get("source", {}) or {}
    author = entry.get("author") or (src.get("title") if isinstance(src, dict) else None)

    return make_document(
        source=source,
        title=title,
        text=text,
        url=link,
        published=published,
        author=author,
    )
