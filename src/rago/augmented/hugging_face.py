""""""

from __future__ import annotations

from typing import Any, Optional

from sentence_transformer import SentenceTransformer

from rago.augmented.base import AugmentedBase
from rago.db import DBBase, FaissDB


class HuggingFaceAug(AugmentedBase):
    model: Any
    k: int = -1
    db: FaissDB
    documents: list[str]

    def __init__(
        self,
        name: str='paraphrase',
        documents: list[str]=[],
        db: FaissDB = FaissDB(),
        k: int = -1,
    ) -> None:
        if name == 'paraphrase':
            self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        else:
            raise Exception(
                "The Augmented class name {name} is not supported."
            )

        self.
        self.k = k

    def search(self, query: str, k: int = -1):
        # Step 1: Generate embeddings for the documents
        document_embeddings = self.model.encode(documents)
        k = k if k > 0 else self.k
