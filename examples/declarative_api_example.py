"""Example of using the Rago declarative API."""

from rago import Rago, Retrieval, Augmentation, Generation, DB, Cache

# Sample data
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

# Build the RAG pipeline using the declarative API
rag = (
    Rago()
    + DB(backend="faiss", top_k=5)
    + Cache(backend="file")
    + Retrieval(backend="string")
    + Augmentation(backend="sentencetransformer", model="all-MiniLM-L6-v2")
    + Generation(
        backend="openai",
        model="gpt-4o-mini",
        prompt_template="Question: {query}\nContext: {context}\nAnswer:",
        api_key="YOUR_OPENAI_API_KEY"  # Replace with your API key or use environment variable
    )
)

# Run the pipeline
result = rag.run(query="What is the fastest animal on Earth?", data=animals_data)
print(result)