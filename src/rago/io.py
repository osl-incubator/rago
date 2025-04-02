"""IO module."""

from __future__ import annotations

from collections import UserDict
from typing import Any

from typeguard import typechecked


@typechecked
class IO(UserDict):
    """Base class for IO."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the IO object."""
        super().__init__(kwargs)

    def __getattr__(self, name: str) -> Any:
        # Only attempt to get the attribute from the underlying data
        # dictionary.
        try:
            return self.data[name]
        except KeyError:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute '{name}'"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        # If setting the "data" attribute itself, bypass our custom logic.
        if name == 'data':
            super().__setattr__(name, value)
        else:
            self.data[name] = value

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({list(self.data.keys())})'


@typechecked
class Input(IO):
    """Represent the input to a pipeline step."""

    def to_output(self) -> Output:
        return Output(**self.data)


@typechecked
class Output(IO):
    """Represent the output to a pipeline step."""

    @staticmethod
    def from_input(inp: Input) -> Output:
        return Output(**inp.data)

    def as_input(self) -> Input:
        return Input(**self.data)
