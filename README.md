# Rago

![CI](https://img.shields.io/github/actions/workflow/status/osl-incubator/rago/main.yaml?logo=github&label=CI)
[![Python Versions](https://img.shields.io/pypi/pyversions/rago)](https://pypi.org/project/rago/)
[![Package Version](https://img.shields.io/pypi/v/rago?color=blue)](https://pypi.org/project/rago/)
![License](https://img.shields.io/pypi/l/rago?color=blue)
[![Discord](https://img.shields.io/discord/796786891798085652?logo=discord&color=blue)](https://opensciencelabs.org/discord)

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

## Roadmap

### 1. Add new Backends

As noted in several GitHub issues, our initial goal is to support as many
backends as possible. This approach will provide valuable insights into user
needs and inform the structure for the next phase.

### 2. Declarative API for Rago

#### Objective

To simplify and streamline the user experience in configuring RAG by introducing
a declarative, composable API—similar to how Plotnine or Altair allows users to
build visualizations.

#### Overview

The current procedural approach in Rago requires users to instantiate and
connect individual components (retrieval, augmentation, generation, etc.)
manually. This can become cumbersome as support for multiple backends grows. We
propose a new declarative interface that lets users define their entire RAG
steps in a single, fluent expression using operator overloading.

#### Proposed Syntax Example

```python
from rago import Rago, Retrieval, Augmentation, Generation, DB, Cache

datasource = ...

rag = (
    Rago()
    + DB(backend="faiss", top_k=5)
    + Cache(backend="file")
    + Retrieval(backend="dummy")
    + Augmentation(backend="openai", model="text-embedding-3-small")
    + Generation(
        backend="openai",
        model="gpt-4o-mini",
        prompt_template="Question: {query}\nContext: {context}\nAnswer:"
    )
)

result = rag.run(query="What is the capital of France?", data=datasource)
```

#### Key Benefits

- **Intuitive Composition:** Users can build complex pipelines by simply adding
  layers together.
- **Modularity:** Each component is encapsulated, making it easy to swap or
  extend backends without altering the overall architecture.
- **Reduced Boilerplate:** The declarative syntax minimizes the need for
  repetitive setup code, focusing on the "what" rather than the "how."
- **Enhanced Readability:** The pipeline’s structure becomes immediately clear,
  promoting easier maintenance and collaboration.

#### Implementation Plan

1. **Define Base Classes:** Develop abstract base classes for each component
   (DB, Cache, Retrieval, Augmentation, Generation) to standardize interfaces
   and facilitate future extensions.
2. **Operator Overloading:** Implement the `__add__` method in the main `Rago`
   class to allow chaining of components, effectively building the pipeline
   through a fluent interface.
3. **Configuration and Defaults:** Integrate sensible defaults and validation
   (using tools like Pydantic) so that users can override only when necessary.
4. **Documentation and Examples:** Provide comprehensive documentation and
   examples to illustrate the new declarative syntax and usage scenarios.

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

from rago import Rago
from rago.generation import LlamaGen
from rago.retrieval import StringRet
from rago.augmented import SentenceTransformerAug

# For Gated LLMs
HF_TOKEN = 'YOUR_HUGGING_FACE_TOKEN'

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
    generation=LlamaGen(api_key=HF_TOKEN),
)
rag.prompt('What is the faster animal on Earth?')
```

### Ollama

For testing the generation with Ollama, run first the following commands:

```bash
$ ollama pull llama3.2:1b
$ ollama serve
```
