"""ChromaDB round-trip test using fake (pre-computed) embeddings - no model
download required. Run:  python -m tests.index_test
"""
from __future__ import annotations

import shutil
import tempfile


def test_vector_store_roundtrip() -> None:
    from src.store.vector_store import VectorStore

    tmp = tempfile.mkdtemp()
    try:
        cfg = {"store": {"persist_dir": tmp, "collection_name": "test_coll"}}
        store = VectorStore(cfg)
        store.reset()
        chunks = [
            {"chunk_id": "a_0", "doc_id": "a", "source": "news",
             "title": "AI chips", "url": "u1", "published": "", "text": "gpu demand"},
            {"chunk_id": "b_0", "doc_id": "b", "source": "company",
             "title": "Earnings", "url": "u2", "published": "", "text": "record revenue"},
            {"chunk_id": "c_0", "doc_id": "c", "source": "hackernews",
             "title": "CUDA", "url": "u3", "published": "", "text": "open toolkit"},
        ]
        embs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        store.add(chunks, embs)
        assert store.count() == 3

        hits = store.query([0.0, 0.9, 0.1], top_k=2)
        assert hits[0]["chunk_id"] == "b_0"          # closest to dim-2 vector
        assert hits[0]["source"] == "company"        # metadata round-trips
        assert "distance" in hits[0]
        print("  ok  test_vector_store_roundtrip (count + nearest + metadata)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_vector_store_roundtrip()
    print("\nChromaDB round-trip test passed.")
