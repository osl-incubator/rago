"""Base classes for retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from typeguard import typechecked

from rago.retrieval.base import RetrievalBase
from rago.retrieval.tools.pdf import extract_text_from_pdf, is_pdf


@typechecked
class FilePathRet(RetrievalBase):
    """File Retrieval class."""

    def _validate(self) -> None:
        """Validate if the source is valid, otherwise raises an exception."""
        if not isinstance(self.source, (str, Path)):
            raise Exception('Argument source should be an string or a Path.')

        source_path = Path(self.source)
        if not source_path.exists():
            raise Exception("File doesn't exist.")


@typechecked
class PDFPathRet(FilePathRet):
    """PDFPathRet Retrieval class."""

    def _validate(self) -> None:
        """Validate if the source is valid, otherwise raises an exception."""
        super()._validate()
        if not is_pdf(self.source):
            raise Exception('Given file is not a PDF.')

    def get(self, query: str = '') -> Iterable[str]:
        """Get the data from the source."""
        text = extract_text_from_pdf(self.source)

        self.logs['text'] = text

        return self.splitter.split(text)
