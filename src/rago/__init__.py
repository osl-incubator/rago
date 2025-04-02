"""Rago."""

from importlib import metadata as importlib_metadata

from rago.augmented import Augmented
from rago.core import Rago
from rago.generation import Generation
from rago.retrieval import Retrieval


def get_version() -> str:
    """Return the program version."""
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return '0.14.4'  # semantic-release


version = get_version()

__version__ = version
__author__ = 'Ivan Ogasawara'
__email__ = 'ivan.ogasawara@gmail.com'

__all__ = [
    'Augmented',
    'Generation',
    'Rago',
    'Retrieval',
    '__author__',
    '__email__',
    '__version__',
]
