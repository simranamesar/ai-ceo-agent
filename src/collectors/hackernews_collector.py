"""Source 3 - tech community signal via the free HN Algolia search API."""
from __future__ import annotations

import requests

from src.collectors.base import BaseCollector
from src.collectors.hn import hn_hit_to_doc
from src.schema import Document


class HackerNewsCollector(BaseCollector):
    source_name = "hackernews"
    config_key = "hackernews"
    API = "https://hn.algolia.com/api/v1/search"

    def collect(self) -> list[Document]:
        docs: list[Document] = []
        cap = self.scfg.get("max_per_query", 40)
        queries = self.scfg.get("queries") or [self.cfg["company"]["name"]]
        for q in queries:
            try:
                resp = requests.get(
                    self.API,
                    params={"query": q, "tags": "story", "hitsPerPage": cap},
                    timeout=20,
                )
                resp.raise_for_status()
                hits = resp.json().get("hits", [])
            except Exception as e:
                print(f"[hackernews] '{q}' failed: {e}")
                continue
            docs.extend(hn_hit_to_doc(h) for h in hits)
        return docs
