"""Source 1 - third-party news via Google News RSS (keyless, high volume)."""
from __future__ import annotations

from urllib.parse import quote_plus

import feedparser

from src.collectors.base import BaseCollector
from src.collectors.rss import rss_entry_to_doc
from src.schema import Document


class NewsCollector(BaseCollector):
    source_name = "news"
    config_key = "news"

    def collect(self) -> list[Document]:
        docs: list[Document] = []
        templates = self.scfg.get("feeds") or [
            "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        ]
        queries = self.scfg.get("queries") or [self.cfg["company"]["name"]]
        cap = self.scfg.get("max_per_query", 40)
        for tmpl in templates:
            for q in queries:
                url = tmpl.format(query=quote_plus(q))
                try:
                    feed = feedparser.parse(url)
                except Exception as e:
                    print(f"[news] '{q}' failed: {e}")
                    continue
                for entry in feed.entries[:cap]:
                    docs.append(rss_entry_to_doc(entry, self.source_name))
        return docs
