"""Provide an extension for caching."""

from __future__ import annotations

from abc import abstractmethod
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib

from typeguard import typechecked

from rago.extensions.base import Extension


@typechecked
class Cache(Extension):
    """Define an extension base for caching."""

    @abstractmethod
    def load(self, key: Any) -> Any:
        """Load the cache for given key."""
        raise Exception(f'Load method is not implemented: {key}')

    @abstractmethod
    def save(self, key: Any, data: Any) -> None:
        """Save the cache for given key."""
        raise Exception(f'Save method is not implemented: {key}')


@typechecked
class CacheFile(Cache):
    """Define a extra step for caching."""

    target_dir: Path

    def __init__(self, target_dir: Path) -> None:
        self.target_dir = target_dir
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def get_file_path(self, key: Any) -> Path:
        """Return the file path."""
        ref = sha256(str(key).encode('utf-8')).hexdigest()
        return self.target_dir / f'{ref}.pkl'

    def load(self, key: Any) -> Any:
        """Load the cache for given key."""
        file_path = self.get_file_path(key)
        if not file_path.exists():
            return
        return joblib.load(file_path)

    def save(self, key: Any, data: Any) -> None:
        """Load the cache for given key."""
        file_path = self.get_file_path(key)
        joblib.dump(data, file_path)
