"""Base Rago classes."""

from __future__ import annotations

from rago.base import ParametersBase, StepBase


class Logs(StepBase):
    """A step that logs pipeline activity."""

    def __init__(self) -> None:
        self.log: list[str] = []

    def process(self, inp: Input) -> Output:
        """Parameter steps do not process data directly."""
        message = f"Logging: received query '{query}' with data '{data}'"
        self.log.append(message)
        logs['log'] = self.log
        return data

    def apply(self, parameters: ParametersBase) -> None:
        # Optionally, adjust log behavior via parameters
        if 'log_prefix' in parameters.params:
            self.log = [parameters.params['log_prefix'] + m for m in self.log]
