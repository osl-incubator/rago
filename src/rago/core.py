"""Rago is Retrieval Augmented Generation lightweight framework."""

from __future__ import annotations

from typing import Any

from typeguard import typechecked

from rago.base import (
    PipelineBase,
)  # Assumed to provide self.stack (list of StepBase instances)
from rago.io import Input, Output


@typechecked
class Rago(PipelineBase):
    """RAG pipeline that composes retrieval, augmentation, and generation."""

    def run(self, query: str, source: Any, device: str = 'auto') -> Output:
        """
        Run the pipeline for a given query and source data.

        Parameters
        ----------
        query : str
            The user query.
        source : Any
            The source data to process (e.g., documents).
        device : str, optional
            The device to use (default 'auto').

        Returns
        -------
        dict[str, Any]
        """
        cur_input = Input(query=query, source=source)
        cur_output = Output.from_input(cur_input)
        for step in self.stack:
            cur_input = cur_output.as_input()
            cur_input.query = query
            cur_output = step.process(cur_input)
        return cur_output
