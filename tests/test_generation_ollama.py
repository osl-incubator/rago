"""Tests for Ollama generation request handling."""

from __future__ import annotations

from typing import Any

from rago.generation.llama import OllamaGen


class _FakeChatResponse:
    """Minimal Ollama chat response stub."""

    class _Message:
        def __init__(self, content: str) -> None:
            self.content = content

    def __init__(self, content: str) -> None:
        self.message = self._Message(content)


class _FakeOllamaClient:
    """Record the latest chat request for assertion."""

    latest_kwargs: dict[str, Any] = {}

    def __init__(self, host: str, headers: dict[str, str]) -> None:
        self.host = host
        self.headers = headers

    def chat(self, **kwargs: Any) -> _FakeChatResponse:
        """Capture the request and return a deterministic response."""
        self.__class__.latest_kwargs = kwargs
        return _FakeChatResponse('Peregrine Falcon')


def _load_fake_ollama(self: OllamaGen) -> None:
    """Replace the Ollama client with a local test double."""
    self._Ollama = _FakeOllamaClient


def test_ollama_generation_forwards_request_options(
    monkeypatch: Any,
) -> None:
    """Pass temperature and token limits through the Ollama chat call."""
    monkeypatch.setattr(OllamaGen, '_load_optional_modules', _load_fake_ollama)

    generator = OllamaGen(
        temperature=0.0001,
        output_max_length=42,
        api_params={
            'base_url': 'http://localhost:11434/',
            'keep_alive': '1m',
            'options': {'top_p': 0.8},
        },
    )

    result = generator.generate('what is the fastest bird?', ['ctx'])

    assert result == 'Peregrine Falcon'
    assert _FakeOllamaClient.latest_kwargs['keep_alive'] == '1m'
    assert _FakeOllamaClient.latest_kwargs['options'] == {
        'top_p': 0.8,
        'temperature': 0.0001,
        'num_predict': 42,
    }
