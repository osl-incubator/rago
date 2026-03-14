"""Declarative configuration helpers for composable Rago pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from typeguard import typechecked

from rago.augmented.db import DBBase
from rago.base import ParametersBase
from rago.extensions.cache import CacheFile
from rago.extensions.logs import Logs as LogsConfig


@typechecked
class Cache(ParametersBase):
    """Resolve a cache backend into step configuration."""

    def __init__(
        self,
        backend: str = 'file',
        target_dir: Path | str = '.rago-cache',
    ) -> None:
        backend_name = backend.lower()
        if backend_name != 'file':
            raise ValueError(f'Unsupported cache backend: {backend}')

        cache = CacheFile(target_dir=target_dir)
        super().__init__(cache=cache)


@typechecked
class DB(ParametersBase):
    """Resolve a vector database backend into step configuration."""

    def __init__(self, backend: str = 'faiss', **kwargs: Any) -> None:
        backend_name = backend.lower()
        db: DBBase

        if backend_name == 'faiss':
            from rago.augmented.db.faiss import FaissDB

            db = FaissDB()
        elif backend_name == 'chroma':
            from rago.augmented.db.chroma import ChromaDB

            db = ChromaDB(**kwargs)
        else:
            raise ValueError(f'Unsupported DB backend: {backend}')

        super().__init__(db=db)


@typechecked
class Logs(LogsConfig):
    """Public alias for attaching logs declaratively."""
