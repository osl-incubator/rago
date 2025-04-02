"""Base classes for retrieval."""

from __future__ import annotations

from typing import Any, Iterable, cast

from typeguard import typechecked

from rago.retrieval.base import RetrievalBase


@typechecked
class StringRet(RetrievalBase):
    """
    String Retrieval class.

    This assumes that the source is already a list of strings.
    """

    def retrieve(self, query: str = '', source: Any = None) -> Iterable[str]:
        """Return the list of strings provided as source."""
        return cast(list[str], source or [])
