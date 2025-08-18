"""RAG Retrieval package."""

from __future__ import annotations

from rago.retrieval.base import RetrievalBase
from rago.retrieval.dummy import StringRet
from rago.retrieval.file import PDFPathRet

__all__ = [
    'PDFPathRet',
    'RetrievalBase',
    'StringRet',
]
