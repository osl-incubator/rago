"""Provide base interfaces."""

from __future__ import annotations

from abc import ABC
from typing import Any, Optional

from rago.extensions.cache import Cache


class RagoBase(ABC):
    """Define base interface for RAG step classes."""

    api_key: str = ''
    cache: Optional[Cache] = None
    logs: dict[str, Any] = {}  # noqa: RUF012

    def __init__(
        self,
        api_key: str = '',
        cache: Optional[Cache] = None,
        logs: dict[str, Any] = {},
    ) -> None:
        self.api_key = api_key
        self.cache = cache
        self.logs = logs

    def _get_cache(self, key: Any) -> Any:
        if not self.cache:
            return
        return self.cache.load(key)

    def _save_cache(self, key: Any, data: Any) -> None:
        if not self.cache:
            return
        self.cache.save(key, data)
