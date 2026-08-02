from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.embeddings import embed
from src.types import RetrievedDocument


def cosine_similarity(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> np.float64:
    """Calculate cosine similarity between two embedding vectors.

    Args:
        a (NDArray[np.float64]): The first embedding vector.
        b (NDArray[np.float64]): The second embedding vector.

    Returns:
        np.float64: The cosine similarity as a NumPy 64-bit floating-point
            scalar.
    """

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(
    query: str,
    documents: Sequence[str],
    doc_embeddings: Sequence[NDArray[np.float64]],
    k: int = 3,
) -> list[RetrievedDocument]:
    """Rank documents by semantic similarity.

    Args:
        query (str): The user's query to embed and compare against the
            documents.
        documents (Sequence[str]): The source documents corresponding to
            ``doc_embeddings``.
        doc_embeddings (Sequence[NDArray[np.float64]]): Precomputed embedding
            vectors for ``documents``.
        k (int): The maximum number of top-ranked results to return.

    Returns:
        list[RetrievedDocument]: Up to ``k`` ranked documents with scores.
    """

    query_emb = embed(query)

    scored_results: list[RetrievedDocument] = []

    for i, emb in enumerate(doc_embeddings):
        score = cosine_similarity(query_emb, emb)

        scored_results.append({
            "score": float(score),
            "document": documents[i],
        })

    # sort by score descending
    scored_results.sort(key=lambda x: x["score"], reverse=True)

    return scored_results[:k]
