"""File-based retrieval implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from typeguard import typechecked

from rago.retrieval.base import RetrievalBase
from rago.retrieval.tools.audio import transcribe_audio
from rago.retrieval.tools.ocr import extract_text_from_image
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


@typechecked
class ImagePathRet(FilePathRet):
    """Image retrieval step using OCR."""

    supported_extensions = {
        '.png',
        '.jpg',
        '.jpeg',
        '.bmp',
        '.tiff',
        '.tif',
        '.webp',
    }

    def _validate(self) -> None:
        super()._validate()

        if self.source is None:
            return

        source_path = Path(self.source)
        if source_path.suffix.lower() not in self.supported_extensions:
            raise Exception(
                'Given file is not a supported image format. '
                f'Supported formats: {sorted(self.supported_extensions)}'
            )

    def retrieve(self, query: str = '', source: Any = None) -> list[str]:
        """Extract OCR text and split it into chunks."""
        del query

        actual_source = self.source if source is None else source
        if actual_source is None:
            raise ValueError('An image source is required for OCR retrieval.')

        text = extract_text_from_image(actual_source)
        return list(self.splitter.split(text))


@typechecked
class AudioPathRet(FilePathRet):
    """Audio retrieval step using speech-to-text."""

    supported_extensions = {
        '.mp3',
        '.wav',
        '.m4a',
        '.flac',
        '.ogg',
        '.aac',
        '.wma',
    }

    def __init__(
        self,
        source: Any = None,
        model_name: str = 'tiny',
        device: str = 'cpu',
        compute_type: str = 'int8',
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        super().__init__(source=source, **kwargs)

    def _validate(self) -> None:
        super()._validate()

        if self.source is None:
            return

        source_path = Path(self.source)
        if source_path.suffix.lower() not in self.supported_extensions:
            raise Exception(
                'Given file is not a supported audio format. '
                f'Supported formats: {sorted(self.supported_extensions)}'
            )

    def retrieve(self, query: str = '', source: Any = None) -> list[str]:
        """Transcribe audio and split the result into chunks."""
        del query

        actual_source = self.source if source is None else source
        if actual_source is None:
            raise ValueError(
                'An audio source is required for audio retrieval.'
            )

        text = transcribe_audio(
            actual_source,
            model_name=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

        return list(self.splitter.split(text))
