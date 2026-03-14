"""Rago core pipeline."""

from __future__ import annotations

from typing import Any

from typeguard import typechecked

from rago.base import Pipeline, StepBase


@typechecked
class Rago(Pipeline):
    """Composable Rago pipeline."""

    def __init__(
        self,
        retrieval: StepBase | None = None,
        augmented: StepBase | None = None,
        generation: StepBase | None = None,
    ) -> None:
        super().__init__()
        for step in (retrieval, augmented, generation):
            if step is not None:
                self | step

    @property
    def logs(self) -> dict[str, dict[str, Any]]:
        """Expose step logs using stable public names when possible."""
        aggregated: dict[str, dict[str, Any]] = {}
        name_counts: dict[str, int] = {}

        for step in self.stack:
            base_name = getattr(
                step, 'log_name', step.__class__.__name__.lower()
            )
            count = name_counts.get(base_name, 0)
            name_counts[base_name] = count + 1
            key = base_name if count == 0 else f'{base_name}_{count + 1}'
            aggregated[key] = getattr(step, 'logs', {})

        return aggregated
