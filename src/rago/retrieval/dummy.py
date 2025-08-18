"""Base classes for retrieval."""

from __future__ import annotations

from typing import Iterable, cast

from typeguard import typechecked

from rago.retrieval.base import RetrievalBase


@typechecked
class StringRet(RetrievalBase):
    """
    String Retrieval class.

    This is a very generic class that assumes that the input (source) is
    already a list of strings.
    """

    def get(self, query: str = '') -> Iterable[str]:
        """Get the data from the sources."""
        return cast(list[str], self.source)
