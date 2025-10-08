"""Provide a declarative cache extension."""

from __future__ import annotations

import logging

from abc import abstractmethod
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib

from typeguard import typechecked

from rago.base import StepBase
from rago.io import Input, Output


@typechecked
class Cache(StepBase):
    """Abstract base class for caching steps in a declarative pipeline."""

    @abstractmethod
    def load(self, key: Any) -> Any:
        """Load cached data for the given key."""
        raise Exception(f'Load method is not implemented for key: {key}')

    @abstractmethod
    def save(self, key: Any, data: Any) -> None:
        """Save data to cache under the given key."""
        raise Exception(f'Save method is not implemented for key: {key}')

    def process(self, inp: Input) -> Output:
        """
        Run the default processing method.

        Compute a key from the query and data,
        attempt to load a cached result, and return it if found. Otherwise,
        return the data unchanged.
        """
        query = inp.query
        data = inp.data

        key = self._compute_key(query, data)
        cached = self.load(key)
        if cached is not None:
            logging.debug(f'Cache hit for key: {key}')
            return cached
        logging.debug(f'No cache found for key: {key}')
        return data

    def _compute_key(self, query: str, data: Any) -> str:
        """Compute a cache key based on the query and data."""
        combined = str(query) + str(data)
        return sha256(combined.encode('utf-8')).hexdigest()


@typechecked
class CacheFile(Cache):
    """File-based cache step that saves and loads data using joblib."""

    target_dir: Path

    def __init__(self, target_dir: Path) -> None:
        self.target_dir = target_dir
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def get_file_path(self, key: Any) -> Path:
        """Return the file path for the given key."""
        ref = sha256(str(key).encode('utf-8')).hexdigest()
        return self.target_dir / f'{ref}.pkl'

    def load(self, key: Any) -> Any:
        """Load the cache for the given key if it exists."""
        file_path = self.get_file_path(key)
        if not file_path.exists():
            return None
        return joblib.load(file_path)

    def save(self, key: Any, data: Any) -> None:
        """Save data to the cache under the given key."""
        file_path = self.get_file_path(key)
        joblib.dump(data, file_path)

    def process(self, inp: Input) -> Output:
        """
        Process the cache for files.

        Compute a cache key and attempt to load cached data. If found, return
        the cached result. Otherwise, save the current data to the cache and
        return it.
        """
        query = inp.query
        data = inp.data

        key = self._compute_key(query, data)
        cached = self.load(key)
        if cached is not None:
            logging.debug(f'Cache hit for key: {key}')
            return cached

        # TODO: fix this
        self.save(key, data)
        logging.debug(f'Cache saved for key: {key}')
        return Output(data=data)
