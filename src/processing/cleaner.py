"""Clean raw text: strip HTML and normalise whitespace."""
from __future__ import annotations

from src.schema import Document
from src.utils import collapse_ws, html_to_text


def clean_text(text: str) -> str:
    return collapse_ws(html_to_text(text))


def clean_documents(docs: list[Document]) -> list[Document]:
    """Clean title/text on every doc; drop empties, fall back to title."""
    out: list[Document] = []
    for d in docs:
        doc = dict(d)
        doc["title"] = collapse_ws(doc.get("title", ""))
        doc["text"] = clean_text(doc.get("text", ""))
        if not doc["text"]:
            doc["text"] = doc["title"]
        if not doc["text"]:
            continue
        out.append(doc)
    return out
