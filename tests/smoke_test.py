"""Offline tests: validate parsing/mapping/cleaning WITHOUT network access.
Run:  python -m tests.smoke_test
"""
from __future__ import annotations

from src.collectors.hn import hn_hit_to_doc
from src.collectors.rss import rss_entry_to_doc
from src.processing.cleaner import clean_documents, clean_text
from src.processing.deduplicator import dedupe_exact


def test_rss_mapping() -> None:
    entry = {
        "title": "NVIDIA posts record revenue",
        "link": "https://example.com/a",
        "published": "Mon, 01 Jan 2026 00:00:00 GMT",
        "summary": "<p>Quarterly <b>revenue</b> jumped.</p>",
        "source": {"title": "Reuters"},
    }
    d = rss_entry_to_doc(entry, "news")
    assert d["source"] == "news"
    assert d["url"] == "https://example.com/a"
    assert d["author"] == "Reuters"
    assert d["id"]
    assert "revenue" in d["text"]


def test_hn_mapping() -> None:
    hit = {
        "title": "NVIDIA open-sources a toolkit",
        "url": "https://example.com/x",
        "author": "pg",
        "points": 120,
        "num_comments": 45,
        "created_at_i": 1735689600,
        "objectID": "42",
    }
    d = hn_hit_to_doc(hit)
    assert d["source"] == "hackernews"
    assert d["metadata"]["points"] == 120
    assert d["published"].startswith("2025")  # 1735689600 -> 2025-01-01 UTC


def test_hn_askhn_fallback_url() -> None:
    d = hn_hit_to_doc({"title": "Ask HN: views on NVDA?", "objectID": "99"})
    assert d["url"].endswith("item?id=99")


def test_clean_text() -> None:
    assert clean_text("<p>Hello    world</p>") == "Hello world"


def test_clean_documents_drops_empty_keeps_title() -> None:
    docs = [
        rss_entry_to_doc({"title": "Title only", "link": "u1"}, "news"),
        rss_entry_to_doc({"title": "", "link": "u2", "summary": "<p></p>"}, "news"),
    ]
    out = clean_documents(docs)
    assert len(out) == 1
    assert out[0]["text"] == "Title only"


def test_dedupe_exact() -> None:
    a = rss_entry_to_doc({"title": "Same", "link": "u"}, "news")
    b = rss_entry_to_doc({"title": "Same", "link": "u"}, "company")
    assert len(dedupe_exact([a, b])) == 1




# ---- stage 3-4 additions (appended) ----

def _cfg():
    return {"processing": {"chunk_size": 50, "chunk_overlap": 10, "min_chunk_chars": 20}}


def test_chunker_short_doc_single_window():
    from src.collectors.rss import rss_entry_to_doc
    from src.processing.chunker import chunk_documents
    d = rss_entry_to_doc({"title": "T", "link": "u", "summary": "short text here"}, "news")
    chunks = chunk_documents([d], _cfg())
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"].endswith("_0")
    assert chunks[0]["doc_id"] == d["id"]
    assert chunks[0]["source"] == "news"


def test_chunker_long_doc_overlaps():
    from src.processing.chunker import chunk_documents
    doc = {"id": "abc", "source": "news", "title": "x", "url": "u",
           "published": "", "text": "A" * 130}
    chunks = chunk_documents([doc], _cfg())
    assert len(chunks) >= 2
    assert all(len(c["text"]) <= 50 for c in chunks)


def test_dedupe_near_drops_similar():
    from src.processing.deduplicator import dedupe_near
    docs = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
    embs = [[1.0, 0.0], [0.999, 0.0447], [0.0, 1.0]]  # 1&2 nearly identical
    kept = dedupe_near(docs, embs, threshold=0.92)
    ids = {d["id"] for d in kept}
    assert ids == {"1", "3"}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} offline tests passed.")
