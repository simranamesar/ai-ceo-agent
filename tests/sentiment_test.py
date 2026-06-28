"""Offline tests for stage 7: label normalisation + aggregation (no model).
Run:  python -m tests.sentiment_test
"""
from __future__ import annotations

from src.intelligence.sentiment import SentimentAnalyzer, _norm_label


def test_label_normalisation() -> None:
    assert _norm_label("POSITIVE") == "positive"
    assert _norm_label("LABEL_0") == "negative"
    assert _norm_label("LABEL_2") == "positive"
    assert _norm_label("Neutral") == "neutral"
    assert _norm_label("negative") == "negative"


def test_aggregate_overall_and_categories() -> None:
    cfg = {"sentiment": {"news_sources": ["news"], "public_sources": ["hackernews"]}}
    sa = SentimentAnalyzer(cfg)
    scored = [
        {"source": "news", "published": "2026-01-01T00:00:00+00:00", "label": "positive", "score": 0.9},
        {"source": "news", "published": "2026-01-01T00:00:00+00:00", "label": "negative", "score": 0.8},
        {"source": "hackernews", "published": "Mon, 02 Jan 2026 12:00:00 GMT", "label": "positive", "score": 0.7},
    ]
    agg = sa.aggregate(scored)
    assert agg["overall"]["total"] == 3
    assert agg["overall"]["net_polarity"] == round((2 - 1) / 3, 3)
    assert agg["by_category"]["news"]["total"] == 2
    assert agg["by_category"]["public"]["positive"] == 1
    # trend has two parseable dates (ISO + RFC-822)
    assert len(agg["trend"]) == 2
    assert agg["trend"][0]["date"] == "2026-01-01"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} stage-7 offline tests passed.")
