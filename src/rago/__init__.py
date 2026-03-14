"""Rago public package interface."""

from __future__ import annotations

from importlib import import_module
from importlib import metadata as importlib_metadata
from typing import Any


def get_version() -> str:
    """Return the installed package version."""
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return '0.14.4'  # semantic-release


version = get_version()

__version__ = version
__author__ = 'Ivan Ogasawara'
__email__ = 'ivan.ogasawara@gmail.com'

__all__ = [
    'DB',
    'Augmented',
    'Cache',
    'Generation',
    'Logs',
    'Rago',
    'Retrieval',
    '__author__',
    '__email__',
    '__version__',
]

_EXPORTS = {
    'Rago': ('rago.core', 'Rago'),
    'Retrieval': ('rago.retrieval', 'Retrieval'),
    'Augmented': ('rago.augmented', 'Augmented'),
    'Generation': ('rago.generation', 'Generation'),
    'Cache': ('rago.config', 'Cache'),
    'DB': ('rago.config', 'DB'),
    'Logs': ('rago.config', 'Logs'),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
