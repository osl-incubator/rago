# Rago

![CI](https://img.shields.io/github/actions/workflow/status/osl-incubator/rago/main.yaml?logo=github&label=CI)
[![Python Versions](https://img.shields.io/pypi/pyversions/rago)](https://pypi.org/project/rago/)
[![Package Version](https://img.shields.io/pypi/v/rago?color=blue)](https://pypi.org/project/rago/)
![License](https://img.shields.io/pypi/l/rago?color=blue)
[![Discord](https://img.shields.io/discord/796786891798085652?logo=discord&color=blue)](https://discord.gg/j95a6Nku44)

Rago is a lightweight framework for RAG.

- Software License: BSD 3 Clause
- Documentation: https://osl-incubator.github.io/rago

## Features

- Vector Database support
    - FAISS
- Retrieval features
    - Support pdf extraction via langchain
- Augmentation (Embedding + Vector Database Search)
    - Support for Sentence Transformer (Hugging Face)
    - Support for Open AI
    - Support for SpaCy
- Generation (LLM)
    - Support for Hugging Face
    - Support for llama (Huggin FAce)
    - Support for OpenAI
    - Support for Gemini

## Installation

If you want to install it for `cpu` only, you can run:

```bash
$ pip install rago[cpu]
```

But, if you want to install it for `gpu` (cuda), you can run:

```bash
$ pip install rago[gpu]
```

## Setup

### Llama 3

In order to use a llama model, visit its page on huggingface and request your
access in its form, for example: https://huggingface.co/meta-llama/Llama-3.2-1B.

After you are granted access to the desired model, you will be able to use it
with Rago.

You will also need to provide a hugging face token in order to download the
models locally, for example:

```python

animals_data = [
    "The Blue Whale is the largest animal ever known to have existed, even "
    "bigger than the largest dinosaurs.",
    "The Peregrine Falcon is renowned as the fastest animal on the planet, "
    "capable of reaching speeds over 240 miles per hour.",
    "The Giant Panda is a bear species endemic to China, easily recognized by "
    "its distinctive black-and-white coat.",
    "The Cheetah is the world's fastest land animal, capable of sprinting at "
    "speeds up to 70 miles per hour in short bursts covering distances up to "
    "500 meters.",
    "The Komodo Dragon is the largest living species of lizard, found on "
    "several Indonesian islands, including its namesake, Komodo.",
]

rag = Rago(
    retrieval=StringRet(animals_data),
    augmented=SentenceTransformerAug(top_k=2),
    generation=LlamaGen(apikey=HF_TOKEN),
)
rag.prompt('What is the faster animal on Earth?')
```
