"""File-based retrieval implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from typeguard import typechecked

from rago.retrieval.base import RetrievalBase
from rago.retrieval.tools.pdf import extract_text_from_pdf, is_pdf


@typechecked
class FilePathRet(RetrievalBase):
    """Base retrieval step for file paths."""

    def _validate(self) -> None:
        if self.source is None:
            return
        if not isinstance(self.source, (str, Path)):
            raise Exception('Argument source should be a string or a Path.')

        source_path = Path(self.source)
        if not source_path.exists():
            raise Exception("File doesn't exist.")


@typechecked
class PDFPathRet(FilePathRet):
    """PDF retrieval step."""

    def _validate(self) -> None:
        super()._validate()
        if self.source is None:
            return
        if not is_pdf(self.source):
            raise Exception('Given file is not a PDF.')

    def retrieve(self, query: str = '', source: Any = None) -> list[str]:
        """Extract and split text from the configured PDF source."""
        del query
        actual_source = self.source if source is None else source
        if actual_source is None:
            raise ValueError('A PDF source is required for PDF retrieval.')

        text = extract_text_from_pdf(actual_source)
        return list(self.splitter.split(text))
