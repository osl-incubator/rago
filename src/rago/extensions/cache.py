"""Cache backends for Rago steps."""

from __future__ import annotations

from abc import abstractmethod
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib

from typeguard import typechecked


@typechecked
class Cache:
    """Abstract cache backend used by pipeline steps."""

    @abstractmethod
    def load(self, key: Any) -> Any:
        """Load cached data for the given key."""

    @abstractmethod
    def save(self, key: Any, data: Any) -> None:
        """Persist data in the cache."""

    def get_file_key(self, key: Any) -> str:
        """Normalize a cache key into a stable string representation."""
        return sha256(str(key).encode('utf-8')).hexdigest()


@typechecked
class CacheFile(Cache):
    """File-based cache backend implemented with joblib."""

    target_dir: Path

    def __init__(self, target_dir: Path | str) -> None:
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def get_file_path(self, key: Any) -> Path:
        """Return the file path for a given cache key."""
        return self.target_dir / f'{self.get_file_key(key)}.pkl'

    def load(self, key: Any) -> Any:
        """Load cached data if present."""
        file_path = self.get_file_path(key)
        if not file_path.exists():
            return None
        return joblib.load(file_path)

    def save(self, key: Any, data: Any) -> None:
        """Persist cached data to disk."""
        file_path = self.get_file_path(key)
        joblib.dump(data, file_path)
