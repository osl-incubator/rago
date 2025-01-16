"""Provide base interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal


class Pipelines(ABC):
    """Define base interface for RAG step classes."""

    # pipeline
    pipelines: dict[str, list[PipelineStep]] = {  # noqa: RUF012
        'pre': [],
        'post': [],
    }

    def set_pipeline(
        self,
        pipeline: list[PipelineStep],
        run_type: Literal['pre', 'post'],
    ) -> None:
        """Set pipeline for given run-type."""
        self.pipelines[run_type] = pipeline

    def run(self, run_type: Literal['pre', 'post']) -> None:
        """Run given run-type pipeline."""
        for step in self.pipelines[run_type]:
            step.run()


class PipelineStep(ABC):
    """Define base interface for RAG step classes."""

    @abstractmethod
    def run(self) -> None:
        """Execute the step."""
        return
