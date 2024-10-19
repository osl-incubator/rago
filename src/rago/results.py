""""""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Result:
    """"""

    idx: int
    step: str
    input: str
    output: str


@dataclass
class Results:
    output: list[ResultItem] = field(default_factory=list)
