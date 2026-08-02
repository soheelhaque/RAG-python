import numpy as np
from numpy.typing import NDArray

from src.embeddings import embed
from src.llm_client import call_llm
from src.prompt_builder import build_prompt
from src.retrieval import retrieve
from src.types import RAGResponse

document_embeddings: dict[tuple[str, ...], list[NDArray[np.float64]]] = {}


def get_document_embeddings(documents: list[str]) -> list[NDArray[np.float64]]:
    """Create and cache embeddings for a corpus when it is first used."""
    corpus = tuple(documents)

    if corpus not in document_embeddings:
        document_embeddings[corpus] = [embed(doc) for doc in documents]

    return document_embeddings[corpus]


def rag(query: str, documents: list[str]) -> RAGResponse:
    """Retrieve relevant context and generate an answer for a user query.

    Args:
        query (str): The user's financial research question.
        documents (list[str]): The corpus selected for the research question.

    Returns:
        RAGResponse: A mapping containing the query, retrieved documents,
            generated prompt, and language-model answer.
    """

    retrieved = retrieve(query, documents, get_document_embeddings(documents), k=3)

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
