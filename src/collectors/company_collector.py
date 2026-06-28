"""Source 2 - the company's own voice: newsroom + official blog RSS."""
from __future__ import annotations

import feedparser

from src.collectors.base import BaseCollector
from src.collectors.rss import rss_entry_to_doc
from src.schema import Document


class CompanyCollector(BaseCollector):
    source_name = "company"
    config_key = "company"

    def collect(self) -> list[Document]:
        docs: list[Document] = []
        cap = self.scfg.get("max_per_feed", 60)
        for url in self.scfg.get("feeds", []):
            try:
                feed = feedparser.parse(url)
            except Exception as e:
                print(f"[company] feed {url} failed: {e}")
                continue
            for entry in feed.entries[:cap]:
                docs.append(rss_entry_to_doc(entry, self.source_name))
        return docs
