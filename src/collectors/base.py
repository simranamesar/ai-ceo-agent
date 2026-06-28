"""Base collector interface. Every source returns a list[Document]."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.schema import Document


class BaseCollector(ABC):
    """Subclass once per source. ``config_key`` selects this source's slice
    of config['collection']['sources']; ``source_name`` tags every document."""

    source_name: str = "base"
    config_key: str = ""

    def __init__(self, config: dict[str, Any]) -> None:
        self.cfg = config
        self.scfg = (
            config.get("collection", {}).get("sources", {}).get(self.config_key, {})
        )

    @abstractmethod
    def collect(self) -> list[Document]:
        """Fetch live items and return them as canonical Documents.

        Implementations must be fully automatic, tag every doc with
        ``source = self.source_name``, and never crash on a bad item/feed.
        """
        raise NotImplementedError
