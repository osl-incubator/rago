"""File-based retrieval classes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, cast

from typeguard import typechecked

from rago.retrieval.base import RetrievalBase
from rago.retrieval.tools.pdf import extract_text_from_pdf, is_pdf


@typechecked
class FilePathRet(RetrievalBase):
    """File Retrieval class."""

    def _validate(self) -> None:
        """Validate that the source is a valid file path."""
        if not isinstance(self.source, (str, Path)):
            raise Exception('Argument source should be a string or a Path.')
        source_path = Path(self.source)
        if not source_path.exists():
            raise Exception("File doesn't exist.")


@typechecked
class PDFPathRet(FilePathRet):
    """PDF Retrieval class."""

    def _validate(self) -> None:
        """Validate that the source is a PDF file."""
        super()._validate()
        if not is_pdf(self.source):
            raise Exception('Given file is not a PDF.')

    def retrieve(self, query: str = '') -> Iterable[str]:
        """Extract text from the PDF, split into chunks, and use caching."""
        cache_key = self.source
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cast(Iterable[str], cached)

        text = extract_text_from_pdf(self.source)
        # self.logs['text'] = text
        result = self.splitter.split(text)
        self._save_cache(cache_key, result)
        return result
