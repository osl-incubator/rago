"""RAG Generation package."""

from __future__ import annotations

from rago.generation.base import GenerationBase
from rago.generation.cohere import CohereGen
from rago.generation.deepseek import DeepSeekGen
from rago.generation.fireworks import FireworksGen
from rago.generation.gemini import GeminiGen
from rago.generation.groq import GroqGen
from rago.generation.hugging_face import HuggingFaceGen
from rago.generation.hugging_face_inf import HuggingFaceInfGen
from rago.generation.llama import LlamaGen, OllamaGen, OllamaOpenAIGen
from rago.generation.openai import OpenAIGen
from rago.generation.phi import PhiGen
from rago.generation.together import TogetherGen

__all__ = [
    'CohereGen',
    'DeepSeekGen',
    'FireworksGen',
    'GeminiGen',
    'GenerationBase',
    'GroqGen',
    'HuggingFaceGen',
    'HuggingFaceInfGen',
    'LlamaGen',
    'OllamaGen',
    'OllamaOpenAIGen',
    'OpenAIGen',
    'PhiGen',
    'TogetherGen',
]
