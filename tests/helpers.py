"""Shared test helpers for provider-backed integration tests."""

from __future__ import annotations

import importlib.util
import socket

from functools import lru_cache, partial
from typing import Any, Callable, TypeVar

import pytest

T = TypeVar('T')

REMOTE_PROVIDER_HOSTS = {
    'cohere': 'api.cohere.com',
    'deepseek': 'api.deepseek.com',
    'fireworks': 'api.fireworks.ai',
    'gemini': 'generativelanguage.googleapis.com',
    'groq': 'api.groq.com',
    'huggingfaceinf': 'api-inference.huggingface.co',
    'openai': 'api.openai.com',
    'together': 'api.together.xyz',
}

UNAVAILABLE_EXCEPTION_NAMES = {
    'APIConnectionError',
    'APITimeoutError',
    'ConnectError',
    'ConnectTimeout',
    'InternalError',
    'InstructorRetryException',
    'RemoteProtocolError',
    'ResponseNotRead',
}

UNAVAILABLE_PATTERNS = (
    'temporary failure in name resolution',
    'connection error',
    'connection refused',
    'all connection attempts failed',
    'failed to establish a new connection',
    'name or service not known',
    'no route to host',
    'network is unreachable',
    'timed out',
    "can't find model",
    'model not found',
    'is not installed',
    'not found in pipeline',
    'connection aborted',
    'unable to open database file',
)


def partial_backend(factory: partial[Any]) -> str:
    """Return the declared backend for a `functools.partial` factory."""
    return str(factory.keywords.get('backend', ''))


@lru_cache(maxsize=None)
def _can_resolve(host: str) -> bool:
    try:
        socket.getaddrinfo(host, 443)
    except OSError:
        return False
    return True


def require_backend_runtime(backend: str) -> None:
    """Skip early when a backend obviously cannot run in this environment."""
    normalized = backend.lower()
    for suffix in ('gen', 'aug'):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break

    host = REMOTE_PROVIDER_HOSTS.get(normalized)
    if host and not _can_resolve(host):
        pytest.skip(f'{backend} backend requires network access.')


def require_spacy_model(model_name: str) -> None:
    """Skip the test when the requested spaCy model is unavailable."""
    if importlib.util.find_spec(model_name) is None:
        pytest.skip(f"spaCy model '{model_name}' is not installed.")


def skip_if_runtime_unavailable(backend: str, exc: Exception) -> None:
    """Skip tests when a provider or local runtime is unavailable."""
    message = str(exc).lower()
    exception_name = exc.__class__.__name__

    if exception_name in UNAVAILABLE_EXCEPTION_NAMES:
        pytest.skip(f'{backend} backend unavailable: {exc}')

    if isinstance(exc, (ConnectionError, OSError, TimeoutError)):
        pytest.skip(f'{backend} backend unavailable: {exc}')

    if any(pattern in message for pattern in UNAVAILABLE_PATTERNS):
        pytest.skip(f'{backend} backend unavailable: {exc}')


def call_or_skip(
    backend: str,
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute a provider-backed call.

    Skip when the required runtime is unavailable.
    """
    require_backend_runtime(backend)
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        skip_if_runtime_unavailable(backend, exc)
        raise
