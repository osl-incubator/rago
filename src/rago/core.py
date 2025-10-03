"""Rago is Retrieval Augmented Generation lightweight framework."""

from __future__ import annotations

from typeguard import typechecked

from rago.base import (
    Pipeline,
)


@typechecked
class Rago(Pipeline):
    """RAG pipeline that composes retrieval, augmentation, and generation."""

    pass
