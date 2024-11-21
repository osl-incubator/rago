"""Rago."""

from importlib import metadata as importlib_metadata

from rago.core import Rago


def get_version() -> str:
    """Return the program version."""
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return '0.9.0'  # semantic-release


version = get_version()

__version__ = version
__author__ = 'Ivan Ogasawara'
__email__ = 'ivan.ogasawara@gmail.com'

__all__ = [
    '__version__',
    '__author__',
    '__email__',
    'Rago',
]
