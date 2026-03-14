"""Pipeline IO objects."""

from __future__ import annotations

from typing import Any

from typeguard import typechecked


@typechecked
class IO(dict[str, Any]):
    """Mapping with attribute access for pipeline inputs and outputs."""

    def __getattr__(self, name: str) -> Any:
        """Return a stored field via attribute access."""
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute '{name}'"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        """Store non-private attributes as mapping entries."""
        if name.startswith('_'):
            super().__setattr__(name, value)
            return
        self[name] = value

    def __repr__(self) -> str:
        """Return a compact debug representation."""
        return f'{self.__class__.__name__}({list(self.keys())})'


@typechecked
class Input(IO):
    """Represent the input to a pipeline step."""

    def to_output(self) -> Output:
        """Convert this input into an output object."""
        return Output(**dict(self))


@typechecked
class Output(IO):
    """Represent the output to a pipeline step."""

    @staticmethod
    def from_input(inp: Input) -> Output:
        """Create an output object from an input object."""
        return Output(**dict(inp))

    def as_input(self) -> Input:
        """Convert this output into an input object."""
        return Input(**dict(self))
