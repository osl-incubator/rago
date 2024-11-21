"""Models used for the unit tests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnimalModel(BaseModel):
    """Model for animals."""

    name: Literal[
        'Blue Whale',
        'Peregrine Falcon',
        'Giant Panda',
        'Cheetah',
        'Komodo Dragon',
        'Arctic Fox',
        'Monarch Butterfly',
        'Great White Shark',
        'Honey Bee',
        'Emperor Penguin',
        'Unknown',
    ] = Field(
        ...,
        description='The predicted class label.',
    )
