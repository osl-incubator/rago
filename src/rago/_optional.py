"""Provide mechanism for optional impports."""

from __future__ import annotations

import importlib

from typing import Any


class OptionalDependencyError(ImportError):
    """OptionalDependencyError class."""

    pass


def require_dependency(
    package: str,
    *,
    extra: str | None = None,
    context: str | None = None,
) -> Any:
    """
    Import `package` lazily.

    If missing, raise a clear error telling how to install.

    Examples
    --------
        tokenizer_lib = require_dependency(
            "sentencepiece", extra="base", context="Tokenizer")
    """
    try:
        return importlib.import_module(package)
    except ImportError as e:
        hint = []
        if extra:
            hint.append(f"Install with: pip install 'rago[{extra}]'")
        else:
            hint.append(f'Install the missing package: pip install {package}')
        if context:
            hint.insert(
                0, f"{context} requires optional dependency '{package}'."
            )
        msg = ' '.join(hint)
        raise OptionalDependencyError(msg) from e
