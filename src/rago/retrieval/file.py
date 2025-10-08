"""File-based retrieval classes."""

from __future__ import annotations

from pathlib import Path

from typeguard import typechecked

from rago.io import Input, Output
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

    def retrieve(self, inp: Input) -> Output:
        """Extract text from the PDF, split into chunks, and use caching."""
        # query = inp.query
        # source = inp.source
        text = extract_text_from_pdf(self.source)
        result = self.splitter.split(text)
        return Output(content=result)

    def process(self, inp: Input) -> Output:
        return self.retrieve(inp.query)
