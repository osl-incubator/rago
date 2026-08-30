"""Audio transcription tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rago._optional import require_dependency


def transcribe_audio(
    file_path: str | Path,
    model_name: str = 'tiny',
    device: str = 'cpu',
    compute_type: str = 'int8',
    **kwargs: Any,
) -> str:
    """Transcribe an audio file using faster-whisper."""
    faster_whisper = require_dependency(
        'faster_whisper',
        extra='audio',
        context='faster-whisper audio transcription',
    )

    model = faster_whisper.WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
    )

    segments, _ = model.transcribe(str(file_path), **kwargs)

    return ' '.join(segment.text.strip() for segment in segments).strip()
