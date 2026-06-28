"""Map a Hacker News Algolia search hit -> Document."""
from __future__ import annotations

from typing import Any

from src.schema import Document, make_document
from src.utils import ts_to_iso


def hn_hit_to_doc(hit: dict[str, Any]) -> Document:
    title = hit.get("title") or hit.get("story_title") or ""
    object_id = hit.get("objectID")
    url = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
    text = title
    if hit.get("story_text"):
        text = f"{title}\n{hit['story_text']}"
    return make_document(
        source="hackernews",
        title=title,
        text=text,
        url=url,
        published=ts_to_iso(hit.get("created_at_i")),
        author=hit.get("author"),
        metadata={
            "points": hit.get("points"),
            "num_comments": hit.get("num_comments"),
            "objectID": object_id,
        },
    )
