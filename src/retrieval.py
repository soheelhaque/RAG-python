import numpy as np

from src.embeddings import embed


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def explain_match(query, doc, score):
    """
    Simple heuristic explanation (NOT LLM-generated)
    Keeps system transparent and deterministic.
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


def retrieve(query, documents, doc_embeddings, k=3):
    query_emb = embed(query)

    scored_results = []

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
