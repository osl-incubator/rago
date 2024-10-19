""""""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class SourceContentBase:
    """SourceContentBase defines the structure for contents."""

    name: str
    source_type: Literal['path', 'string', 'other']
    chunk_type: Literal['sentence', 'paragraphs', 'size', 'other']


@dataclass
class RetrievalSourceBase:
    """RetrievalSourceBase defines the bases for sources."""

    def get(self) -> SourceContentBase: ...


class RetrievalBase:
    """Base Retrieval class."""

    sources: RetrievalSourceBase
    content: Any

    def __init__(self, sources: list[RetrievalSourceBase]) -> None:
        self.sources = sources

    # def run(self.)


class StringSourceContent(SourceContentBase): ...


class StringRet(RetrievalBase):
    """String Retrieval class."""

    ...
