"""In-memory retrieval implementations."""

from __future__ import annotations

from typing import Any, cast

from typeguard import typechecked

from rago.base import ensure_list
from rago.retrieval.base import RetrievalBase


@typechecked
class StringRet(RetrievalBase):
    """Retrieval step for sources that are already plain text chunks."""

    def retrieve(self, query: str = '', source: Any = None) -> list[str]:
        """Return the provided string chunks unchanged."""
        del query
        value = self.source if source is None else source
        return cast(list[str], ensure_list(value))
