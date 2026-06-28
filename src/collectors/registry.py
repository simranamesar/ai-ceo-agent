"""Build the enabled collectors from config and run them all.

Saves per-source raw JSON + a combined, exact-deduped documents.json, and
checks the hard minimums (>=100 docs, >=3 independent sources).
"""
from __future__ import annotations

import os
from typing import Any

from src.collectors.company_collector import CompanyCollector
from src.collectors.hackernews_collector import HackerNewsCollector
from src.collectors.news_collector import NewsCollector
from src.processing.deduplicator import dedupe_exact
from src.schema import Document
from src.utils import save_json

TYPE_MAP = {
    "google_news": NewsCollector,
    "rss": CompanyCollector,
    "hackernews": HackerNewsCollector,
}


def build_collectors(cfg: dict[str, Any]) -> list[Any]:
    collectors = []
    for name, scfg in cfg["collection"]["sources"].items():
        if not scfg.get("enabled"):
            continue
        cls = TYPE_MAP.get(scfg.get("type"))
        if cls is None:
            print(f"[warn] source '{name}' has unknown type '{scfg.get('type')}'")
            continue
        collectors.append(cls(cfg))
    return collectors


def collect_all(cfg: dict[str, Any]) -> list[Document]:
    raw_dir = cfg["paths"]["raw"]
    per_source: dict[str, int] = {}
    all_docs: list[Document] = []

    for collector in build_collectors(cfg):
        try:
            docs = collector.collect()
        except Exception as e:
            print(f"[error] collector '{collector.source_name}' failed: {e}")
            docs = []
        per_source[collector.source_name] = len(docs)
        save_json(docs, os.path.join(raw_dir, f"{collector.source_name}.json"))
        all_docs.extend(docs)

    all_docs = dedupe_exact(all_docs)
    save_json(all_docs, os.path.join(raw_dir, "documents.json"))

    n_sources = sum(1 for v in per_source.values() if v > 0)
    print(f"[collect] per-source counts: {per_source}")
    print(f"[collect] {len(all_docs)} unique docs across {n_sources} live sources")

    if len(all_docs) < cfg["collection"]["min_documents"]:
        print(f"[warn] below min_documents "
              f"({len(all_docs)}/{cfg['collection']['min_documents']}) "
              f"- add queries/feeds")
    if n_sources < cfg["collection"]["min_sources"]:
        print(f"[warn] below min_sources "
              f"({n_sources}/{cfg['collection']['min_sources']})")
    return all_docs
