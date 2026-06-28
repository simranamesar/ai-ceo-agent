"""Offline tests for stage 5: JSON extraction, RAG context, evidence, and the
engine end-to-end with a STUBBED LLM + RAG (no network, no model download).
Run:  python -m tests.intel_test
"""
from __future__ import annotations

from src.agent.rag_chain import RagChain
from src.intelligence.engine import IntelligenceEngine
from src.intelligence.evidence import build_evidence
from src.llm.client import extract_json


def test_extract_json_with_fences_and_think() -> None:
    raw = '<think>let me reason</think>\nSure!\n```json\n[{"title": "x"}]\n```'
    assert extract_json(raw) == [{"title": "x"}]


def test_extract_json_array_of_objects_with_braces() -> None:
    raw = 'prose [{"a": {"b": 1}}, {"c": [1,2]}] trailing'
    assert extract_json(raw) == [{"a": {"b": 1}}, {"c": [1, 2]}]


def test_build_context_is_numbered() -> None:
    ctx = RagChain.build_context(
        [{"source": "news", "title": "T", "text": "body", "url": "u"}]
    )
    assert ctx.startswith("[1] (news) T")
    assert "URL: u" in ctx


def test_build_evidence_dedupes_by_doc() -> None:
    chunks = [
        {"doc_id": "d1", "source": "news", "title": "A", "url": "u1", "text": "x" * 400},
        {"doc_id": "d1", "source": "news", "title": "A", "url": "u1", "text": "y"},
        {"doc_id": "d2", "source": "company", "title": "B", "url": "u2", "text": "z"},
    ]
    ev = build_evidence(chunks)
    assert len(ev) == 2
    assert len(ev[0]["snippet"]) == 300  # trimmed


class _FakeRag:
    """Returns fixed chunks and the real numbered context formatter."""

    CHUNKS = [
        {"chunk_id": "d1_0", "doc_id": "d1", "source": "news", "title": "AI demand",
         "url": "u1", "published": "", "text": "datacenter GPU demand surging", "distance": 0.1},
        {"chunk_id": "d2_0", "doc_id": "d2", "source": "company", "title": "Q3",
         "url": "u2", "published": "", "text": "record revenue", "distance": 0.2},
    ]

    def retrieve(self, query):
        return list(self.CHUNKS)

    @staticmethod
    def build_context(chunks):
        return RagChain.build_context(chunks)


class _FakeLLM:
    """Returns canned findings that cite context source [1]."""

    def json_chat(self, system, user):
        return [{"title": "Expand DC GPUs", "description": "ride demand",
                 "impact": "high", "evidence": [1]}]


def test_engine_attaches_real_evidence_and_confidence() -> None:
    cfg = {"company": {"name": "NVIDIA"}, "intelligence": {"top_k": 5}}
    engine = IntelligenceEngine(cfg, _FakeRag(), _FakeLLM())
    opps = engine.extract_opportunities()
    assert len(opps) == 1
    item = opps[0]
    # evidence index [1] -> first chunk (doc d1)
    assert item["evidence"][0]["url"] == "u1"
    assert item["evidence"][0]["source"] == "news"
    # confidence derived from retrieval distance (1 - 0.1)
    assert item["confidence"] == 0.9


def test_salvage_recovers_truncated_array() -> None:
    from src.llm.client import _salvage_objects
    truncated = '[{"title": "A", "x": 1}, {"title": "B", "y": 2}, {"title": "C", "z'
    out = _salvage_objects(truncated)
    assert [o["title"] for o in out] == ["A", "B"]  # two complete objects recovered


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} stage-5 offline tests passed.")
