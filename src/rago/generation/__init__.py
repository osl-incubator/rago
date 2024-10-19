"""RAG Generation package."""

from __future__ import annotations

from rago.generation.base import GenerationBase
from rago.generation.hugging_face import HuggingFaceGen

__all__ = [
    'GenerationBase',
    'HuggingFaceGen',
]
