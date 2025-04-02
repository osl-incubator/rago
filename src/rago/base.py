"""Base Rago classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import UserDict
from typing import Any, List, Self

from rago.io import Input, Output


class RagBase(ABC): ...


class StepBase(ABC):
    """Abstract base for all steps in the pipeline."""

    params: ParametersBase

    def _load_optional_modules(self) -> None:
        """Load optional modules."""
        ...

    def _save_cache(self, key: Any, data: Any) -> None:
        if not self.cache:
            return
        self.cache.save(key, data)
    @abstractmethod
    def process(self, inp: Input) -> Output:
        """Process the query and data, updating logs as needed."""
        pass

    def apply(self, parameters: ParametersBase) -> None:
        """Apply additional parameters. Default is no-op."""
        pass


class ParametersBase(UserDict, StepBase):
    """Base class for parameter steps."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(kwargs)

    def process(self, inp: Input) -> Output:
        """Parameter steps do not process data directly."""
        self.apply(inp)
        return inp.to_output()

    def apply(self, inp: Input) -> None:
        """Apply the parameters."""
        self.data.update(inp)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.data})'


class PipelineBase(ABC):
    """Base pipeline class that can be built declaratively."""

    def __init__(self) -> None:
        self.stack: List[StepBase] = []
        self.global_params: List[ParametersBase] = []

    def __add__(self, other: StepBase) -> Self:
        """Overload the '+' operator to add a new step to the pipeline.

        If a ParametersBase is added when no steps exist, store it globally.
        Otherwise, apply global parameters to the new step.
        """
        if isinstance(other, ParametersBase):
            if not self.stack:
                self.global_params.append(other)
            else:
                # Apply parameters to the last step
                self.stack[-1].apply(other)
            return self
        else:
            # Apply all global parameters to the step before adding it
            for params in self.global_params:
                other.apply(params)
            self.stack.append(other)
            return self

    @abstractmethod
    def run(self, query: str, data: Any) -> Any:
        """Run the pipeline given a query and some data."""
        pass
