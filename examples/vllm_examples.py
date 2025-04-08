"""Example script demonstrating the use of vLLM with Rago."""

from rago import Rago
from rago.generation import VllmGen
from rago.retrieval import StringRet
from rago.augmented import SentenceTransformerAug
from rago.augmented.db import FaissDB

# Sample data
datasource = [
    "The Blue Whale is the largest animal ever known to have existed, even bigger than the largest dinosaurs.",
    "The Peregrine Falcon is renowned as the fastest animal on the planet, capable of reaching speeds over 240 miles per hour.",
    "The Cheetah can accelerate from 0 to 60 miles per hour in just 3 seconds, making it the fastest land animal.",
    "Elephants have the longest gestation period of any mammal, carrying their young for up to 22 months before giving birth.",
    "Humans are the only species known to use complex language with grammar and syntax.",
    "Octopuses have three hearts, nine brains, and blue blood.",
    "Tardigrades (water bears) can survive in space, extreme temperatures, and without water for decades.",
    "Axolotls can regenerate entire limbs, parts of their brain, and even their heart."
]

def main():
    # Initialize components
    retrieval = StringRet(source=datasource)
    
    augmented = SentenceTransformerAug(
        model_name="all-MiniLM-L6-v2",
        db=FaissDB(),
        top_k=3
    )
    
    # Initialize vLLM generator
    # Replace with your HuggingFace token if using gated models
    vllm_gen = VllmGen(
        model_name="meta-llama/Llama-3.2-1B",  # Or any other model supported by vLLM
        temperature=0.7,
        output_max_length=256,
        device="cuda",  # Use "cpu" if no GPU is available
        prompt_template=(
            "Answer the following question using the provided context:\n"
            "Question: {query}\n"
            "Context: {context}\n"
            "Answer:"
        )
    )
    
    # Create RAG pipeline
    rag = Rago(
        retrieval=retrieval,
        augmented=augmented,
        generation=vllm_gen
    )
    
    # Run queries
    queries = [
        "What is the fastest animal on Earth?",
        "Which animal has unique regenerative abilities?",
        "Tell me about animals with unusual anatomical features.",
    ]
    
    for query in queries:
        print(f"\n\nQuery: {query}")
        print("-" * 50)
        result = rag.prompt(query)
        print(f"Result: {result}")
        
        # Show which context pieces were used
        print("\nContext used:")
        for i, ctx in enumerate(rag.logs["augmented"]["result"]):
            print(f"{i+1}. {ctx[:100]}...")

if __name__ == "__main__":
    main()