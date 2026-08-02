import numpy as np
from numpy.typing import NDArray

from data.financial_docs_short import documents
from src.embeddings import embed
from src.llm_client import call_llm
from src.prompt_builder import build_prompt
from src.retrieval import retrieve
from src.types import RAGResponse

doc_embeddings: list[NDArray[np.float64]] | None = None


def get_document_embeddings() -> list[NDArray[np.float64]]:
    """Create and cache corpus embeddings when the pipeline first runs."""
    global doc_embeddings

    if doc_embeddings is None:
        doc_embeddings = [embed(doc) for doc in documents]

    return doc_embeddings


def rag(query: str) -> RAGResponse:
    """Retrieve relevant context and generate an answer for a user query.

    Args:
        query (str): The user's financial research question.

    Returns:
        RAGResponse: A mapping containing the query, retrieved documents,
            generated prompt, and language-model answer.
    """

    retrieved = retrieve(query, documents, get_document_embeddings(), k=3)

    # optional debug print (VERY useful for learning)
    print("\n--- RETRIEVAL DEBUG ---")
    for i, r in enumerate(retrieved, 1):
        print(f"\nRank {i}")
        print(f"Score: {r['score']:.4f}")
        print(f"Explanation: {r['explanation']}")
        print(f"Doc: {r['document'][:120]}...")
    print("\n")

    prompt = build_prompt(query, retrieved)
    answer = call_llm(prompt)

    return {"query": query, "retrieved_docs": retrieved, "prompt": prompt, "answer": answer}
