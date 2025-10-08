"""Declarative Retrieval API for Rago."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, cast

from typeguard import typechecked

from rago.base import ParametersBase, StepBase
from rago.io import Input, Output

# Import specialized retrieval implementations
from rago.retrieval.base import RetrievalBase
from rago.retrieval.dummy import StringRet

# from rago.retrieval.file import PDFPathRet


@typechecked
@dataclass
class RetrievalParameters(ParametersBase):
    """Parameters for configuring retrieval steps."""

    source: Any = None
    api_key: str = ''
    api_params: Dict[str, Any] = field(default_factory=dict)

    def process(self, inp: Input) -> Output:
        data = inp.get('data')
        return data

    def apply(self, parameters: RetrievalParameters) -> None:
        self.params.update(parameters.params)

    def __repr__(self) -> str:
        return f'RetrievalParameters({self.__dict__.keys()})'


@typechecked
class Retrieval(StepBase):
    """
    Public Retrieval class for Rago.

    Users instantiate Retrieval with desired configuration parameters such as:
      - source: the data source (e.g., a list of strings or a file path)
      - backend: the retrieval backend to use ("string", "pdf", etc.)

    When get() or process() is called, the proper specialized retrieval instance
    is lazily resolved.
    """

    def __init__(
        self,
        source: Any = None,
        backend: str = 'string',
        api_key: str = '',
        api_params: Dict[str, Any] = {},
    ) -> None:
        self.params = RetrievalParameters(
            source=source,
            api_key=api_key,
            api_params=api_params,
        )

        self.backend = backend

    def __call__(self, **kwargs: Any) -> Retrieval:
        params = RetrievalParameters(**kwargs)
        self.apply(params)
        return self

    def apply(self, parameters: RetrievalParameters) -> None:
        for key, value in parameters.params.items():
            setattr(self.params, key, value)

    def _resolve(self) -> RetrievalBase:
        """Resolve and return the specialized retrieval instance."""
        common_params = deepcopy(self.params.__dict__)

        if self.backend == 'string':
            return StringRet(**common_params)
        elif self.backend == 'pdf':
            # Lazy import for PDFPathRet.
            from rago.retrieval.file import PDFPathRet

            return PDFPathRet(**common_params)
        else:
            raise Exception(f'Unsupported retrieval backend: {self.backend}')

    def retrieve(self, inp: Input) -> Iterable[str]:
        """Delegate get() to the specialized retrieval instance."""
        retrieval_instance = self._resolve()
        return retrieval_instance.retrieve(inp)

    def process(self, inp: Input) -> Output:
        """Process the retrieval step."""
        return self.retrieve(inp)
