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


def explain_match(query: str, doc: str, score: float) -> str:
    """Explain a document match using a deterministic heuristic.

    Args:
        query (str): The user's query.
        doc (str): The document being evaluated.
        score (float): The document's cosine-similarity score.

    Returns:
        str: A human-readable description of semantic strength and keyword
            overlap.
    """

    if score > 0.85:
        strength = "very strong semantic match"
    elif score > 0.75:
        strength = "strong semantic match"
    elif score > 0.65:
        strength = "moderate semantic match"
    else:
        strength = "weak semantic similarity"

    # crude keyword overlap signal (optional but useful for intuition)
    query_terms = set(query.lower().split())
    doc_terms = set(doc.lower().split())
    overlap = query_terms.intersection(doc_terms)

    overlap_text = (
        f"Keyword overlap detected: {', '.join(list(overlap)[:5])}"
        if overlap else
        "No direct keyword overlap (semantic match only)"
    )

    return f"{strength}. {overlap_text}"


def retrieve(
    query: str,
    documents: Sequence[str],
    doc_embeddings: Sequence[NDArray[np.float64]],
    k: int = 3,
) -> list[RetrievedDocument]:
    """Rank documents by semantic similarity and explain each result.

    Args:
        query (str): The user's query to embed and compare against the
            documents.
        documents (Sequence[str]): The source documents corresponding to
            ``doc_embeddings``.
        doc_embeddings (Sequence[NDArray[np.float64]]): Precomputed embedding
            vectors for ``documents``.
        k (int): The maximum number of top-ranked results to return.

    Returns:
        list[RetrievedDocument]: Up to ``k`` ranked documents with scores and
            match explanations.
    """

    query_emb = embed(query)

    scored_results: list[RetrievedDocument] = []

    for i, emb in enumerate(doc_embeddings):
        score = cosine_similarity(query_emb, emb)

        scored_results.append({
            "score": float(score),
            "document": documents[i],
            "explanation": None  # filled after sorting
        })

    # sort by score descending
    scored_results.sort(key=lambda x: x["score"], reverse=True)

    # add explanations after ranking
    for r in scored_results:
        r["explanation"] = explain_match(query, r["document"], r["score"])

    return scored_results[:k]
