## Using vLLM with Rago

Rago now includes integration with [vLLM](https://github.com/vllm-project/vllm), a high-performance library for LLM inference and serving.

### Basic Usage

```python
from rago import Rago
from rago.generation import VllmGen
from rago.retrieval import StringRet
from rago.augmented import SentenceTransformerAug

# Sample data source
datasource = [
    "The Blue Whale is the largest animal ever known to have existed, even bigger than the largest dinosaurs.",
    "The Peregrine Falcon is renowned as the fastest animal on the planet, capable of reaching speeds over 240 miles per hour.",
    "The Cheetah can accelerate from 0 to 60 miles per hour in just 3 seconds, making it the fastest land animal.",
    "Elephants have the longest gestation period of any mammal, carrying their young for up to 22 months before giving birth."
]

# Create the vLLM generator
vllm_gen = VllmGen(
    model_name="meta-llama/Llama-3.2-1B",  # Specify your preferred model
    temperature=0.7,
    output_max_length=256,
    device="cuda",  # Use "cpu" if no GPU is available
    api_key="hf_your_huggingface_token",  # Only needed for gated models
    prompt_template="Answer the following question using the provided context:\nQuestion: {query}\nContext: {context}\nAnswer:"
)

# Create Rago instance with vLLM
retrieval = StringRet(source=datasource)
augmented = SentenceTransformerAug()
rag = Rago(
    retrieval=retrieval,
    augmented=augmented,
    generation=vllm_gen
)

# Run a query
result = rag.prompt("What is the fastest animal on Earth?")
print(result)