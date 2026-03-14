"""Base classes for composable Rago pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Iterable, Mapping
from typing import Any

try:
    from typing import Self
except ImportError:  # pragma: no cover - Python < 3.11
    from typing_extensions import Self

from typeguard import typechecked

from rago.io import Input, Output


def _is_cache_backend(value: Any) -> bool:
    return hasattr(value, 'load') and hasattr(value, 'save')


def _is_vector_db(value: Any) -> bool:
    return hasattr(value, 'embed') and hasattr(value, 'search')


def _is_text_splitter(value: Any) -> bool:
    return hasattr(value, 'split') and not isinstance(value, (bytes, str))


def config_to_dict(parameters: Any) -> dict[str, Any]:
    """Normalize step configuration objects into a plain dictionary."""
    if parameters is None:
        return {}
    if isinstance(parameters, ParametersBase):
        return dict(parameters.params)
    if isinstance(parameters, Mapping):
        return dict(parameters)
    return {}


def ensure_list(value: Any) -> list[Any]:
    """Normalize scalar or iterable values into a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


@typechecked
class ParametersBase(UserDict[str, Any]):
    """Base class for declarative step configuration."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(kwargs)

    @property
    def params(self) -> dict[str, Any]:
        """Expose the underlying parameter mapping."""
        return self.data

    def __getattr__(self, name: str) -> Any:
        """Return a parameter value via attribute access."""
        try:
            return self.data[name]
        except KeyError as exc:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute '{name}'"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        """Store non-internal attributes in the parameter mapping."""
        if name == 'data':
            super().__setattr__(name, value)
            return
        self.data[name] = value

    def apply(self, parameters: Any) -> None:
        """Merge additional configuration into this object."""
        self.data.update(config_to_dict(parameters))

    def process(self, inp: Input) -> Output:
        """Return the input unchanged for configuration-only objects."""
        return inp.to_output()

    def __repr__(self) -> str:
        """Return a compact debug representation."""
        return f'{self.__class__.__name__}({self.data})'


class PipelineBase(ABC):
    """Base pipeline class that can be built declaratively."""

    def __init__(self) -> None:
        self.stack: list[StepBase] = []
        self.global_params: list[Any] = []

    def __or__(self, other: Any) -> Self:
        """Append a step, merge a pipeline, or attach configuration."""
        if isinstance(other, PipelineBase):
            for params in other.global_params:
                self | params
            for step in other.stack:
                self | step
            return self

        if isinstance(other, StepBase):
            for params in self.global_params:
                other.apply(params)
            self.stack.append(other)
            return self

        if not self.stack:
            self.global_params.append(other)
        else:
            self.stack[-1].apply(other)
        return self

    @abstractmethod
    def run(
        self,
        query: str = '',
        source: Any = None,
        data: Any = None,
        **kwargs: Any,
    ) -> Output:
        """Run the pipeline."""

    def process(self, inp: Input) -> Output:
        """Run this pipeline when embedded as a step."""
        source = inp.get('source')
        data = inp.get('data', inp.get('content', source))
        extra = {
            key: value
            for key, value in inp.items()
            if key not in {'content', 'data', 'query', 'source'}
        }
        return self.run(inp.query, source=source, data=data, **extra)


class StepBase(ABC):
    """Abstract base for all pipeline steps."""

    log_name: str = 'step'

    def __init__(self) -> None:
        self.cache: Any = None
        self.logs: dict[str, Any] = {}

    def __or__(self, other: Any) -> Pipeline:
        """Create a pipeline that starts with this step."""
        pipeline = Pipeline()
        pipeline | self
        pipeline | other
        return pipeline

    def _load_optional_modules(self) -> None:
        """Load optional dependencies for this step."""

    def _get_cache(self, key: Any) -> Any:
        if not self.cache:
            return None
        return self.cache.load(key)

    def _save_cache(self, key: Any, data: Any) -> None:
        if not self.cache:
            return
        self.cache.save(key, data)

    @abstractmethod
    def process(self, inp: Input) -> Output:
        """Process the current pipeline input."""

    def apply(self, parameters: Any) -> None:
        """Apply attached configuration to the step."""
        if parameters is None:
            return

        if _is_cache_backend(parameters):
            self.cache = parameters
            return

        if _is_vector_db(parameters):
            setattr(self, 'db', parameters)
            return

        if _is_text_splitter(parameters):
            setattr(self, 'splitter', parameters)
            return

        for key, value in config_to_dict(parameters).items():
            if key == 'cache':
                self.cache = value
            elif key == 'logs':
                self.logs = value or {}
            else:
                setattr(self, key, value)


@typechecked
class Pipeline(PipelineBase):
    """Composable pipeline of arbitrary Rago steps."""

    def run(
        self,
        query: str = '',
        source: Any = None,
        data: Any = None,
        **kwargs: Any,
    ) -> Output:
        """Run all configured steps for the given query and source."""
        if source is None and data is not None:
            source = data

        content = kwargs.pop('content', None)
        if content is None:
            content = data if data is not None else source

        cur_output = Output(
            query=query,
            source=source,
            data=content,
            content=content,
            **kwargs,
        )

        for step in self.stack:
            cur_input = cur_output.as_input()
            cur_input.query = query
            cur_output = step.process(cur_input)

        return cur_output

    def prompt(
        self,
        query: str,
        source: Any = None,
        data: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run the pipeline and return the primary result value."""
        output = self.run(query=query, source=source, data=data, **kwargs)
        if 'result' in output:
            return output.result
        if 'content' in output:
            return output.content
        return output
