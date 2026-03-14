"""Logging configuration helpers for Rago pipelines."""

from __future__ import annotations

from typing import Any

from typeguard import typechecked

from rago.base import ParametersBase


@typechecked
class Logs(ParametersBase):
    """Attach a mutable log dictionary to a step."""

    def __init__(self, target: dict[str, Any] | None = None) -> None:
        super().__init__(logs=target if target is not None else {})
