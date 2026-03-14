"""Support langchain text splitter."""

from __future__ import annotations

from importlib import import_module
from typing import List, cast

try:
    _langchain_text_splitters = import_module('langchain_text_splitters')
except ImportError:  # pragma: no cover - compatibility with older LangChain
    _langchain_text_splitters = import_module('langchain.text_splitter')

from rago.retrieval.text_splitter.base import TextSplitterBase


class LangChainTextSplitter(TextSplitterBase):
    """LangChain Text Splitter class."""

    default_splitter_name: str = 'RecursiveCharacterTextSplitter'

    def _validate(self) -> None:
        """Validate if the initial parameters are valid."""
        valid_splitter_names = ['RecursiveCharacterTextSplitter']

        if self.splitter_name not in valid_splitter_names:
            raise Exception(
                f'The given splitter_name {self.splitter_name} is not valid. '
                f'Valid options are {valid_splitter_names}'
            )

    def _setup(self) -> None:
        """Set up the object according to the given parameters."""
        if self.splitter_name == 'RecursiveCharacterTextSplitter':
            self.splitter = (
                _langchain_text_splitters.RecursiveCharacterTextSplitter
            )

    def split(self, text: str) -> list[str]:
        """Split text into smaller chunks for processing."""
        text_splitter = self.splitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=True,
        )
        return cast(List[str], text_splitter.split_text(text))
