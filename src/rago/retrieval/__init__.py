"""Composable retrieval APIs for Rago."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from typeguard import typechecked

from rago.base import ParametersBase, StepBase, config_to_dict
from rago.io import Input, Output
from rago.retrieval.base import RetrievalBase
from rago.retrieval.text_splitter import LangChainTextSplitter

__all__ = [
    'PDFPathRet',
    'Retrieval',
    'RetrievalBase',
    'RetrievalParameters',
    'StringRet',
]


@typechecked
class RetrievalParameters(ParametersBase):
    """Parameters for configuring retrieval steps."""


@typechecked
class Retrieval(StepBase):
    """Public retrieval wrapper that resolves a concrete backend lazily."""

    log_name = 'retrieval'

    def __init__(
        self,
        source: Any = None,
        backend: str = 'string',
        api_key: str = '',
        api_params: dict[str, Any] | None = None,
        splitter: Any = None,
        cache: Any = None,
        logs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend.lower()
        self.params = RetrievalParameters(
            source=source,
            api_key=api_key,
            api_params=api_params or {},
        )
        self.splitter = splitter or LangChainTextSplitter(
            'RecursiveCharacterTextSplitter'
        )
        self.cache = cache
        self.logs = logs or {}

    def __call__(self, **kwargs: Any) -> Retrieval:
        """Update this wrapper with additional retrieval parameters."""
        self.apply(RetrievalParameters(**kwargs))
        return self

    def apply(self, parameters: Any) -> None:
        """Apply declarative configuration to the retrieval wrapper."""
        super().apply(parameters)
        for key, value in config_to_dict(parameters).items():
            if key == 'backend' and isinstance(value, str):
                self.backend = value.lower()
            elif key == 'splitter':
                self.splitter = value
            else:
                self.params.params[key] = value

    def _resolve(self, source: Any = None) -> RetrievalBase:
        config = deepcopy(self.params.params)
        if config.get('source') is None and source is not None:
            config['source'] = source
        if self.splitter is not None:
            config['splitter'] = self.splitter
        if self.cache is not None:
            config['cache'] = self.cache
        if self.logs:
            config['logs'] = self.logs

        if self.backend == 'string':
            from rago.retrieval.dummy import StringRet

            return StringRet(**config)
        if self.backend == 'pdf':
            from rago.retrieval.file import PDFPathRet

            return PDFPathRet(**config)
        raise Exception(f'Unsupported retrieval backend: {self.backend}')

    def retrieve(self, query: str = '', source: Any = None) -> list[str]:
        """Resolve the concrete retriever and fetch content."""
        retrieval_instance = self._resolve(source=source)
        return retrieval_instance.retrieve(query=query, source=source)

    def get(self, query: str = '', source: Any = None) -> list[str]:
        """Backward-compatible alias for `retrieve`."""
        return self.retrieve(query=query, source=source)

    def process(self, inp: Input) -> Output:
        """Process the current pipeline source with retrieval."""
        source = self.params.params.get('source')
        if source is None:
            source = inp.get('source', inp.get('content'))

        result = self.retrieve(query=inp.query, source=source)
        output = Output.from_input(inp)
        output.content = result
        output.data = result
        return output


def __getattr__(name: str) -> Any:
    if name == 'StringRet':
        from rago.retrieval.dummy import StringRet

        return StringRet
    if name == 'PDFPathRet':
        from rago.retrieval.file import PDFPathRet

        return PDFPathRet
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
