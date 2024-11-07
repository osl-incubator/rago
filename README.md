# Rago

Rago is a lightweight framework for RAG.

- Software License: BSD 3 Clause
- Documentation: https://osl-incubator.github.io/rago

## Features

- Support for Hugging Face
- Support for llama

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

you will also need to provide a hugging face token in order to download the
models locally, for example:

```python

rag = Rago(
    retrieval=StringRet(animals_data),
    augmented=SentenceTransformerAug(top_k=3),
    generation=LlamaGen(apikey=HF_TOKEN),
)
rag.prompt('Is there any animals larger than a dinosaur?')
```
