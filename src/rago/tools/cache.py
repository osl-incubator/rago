"""Provide tools for cache."""

from __future__ import annotations

from typeguard import typechecked

from rago.base import PipelineStep


@typechecked
class Cache(PipelineStep):
    """Define a extra step for caching."""

    def run(self) -> None:
        """Execute the step."""
        return
