"""Offline tests for the baseline retrieval helpers."""

import numpy as np
from numpy.typing import NDArray

from src import retrieval


def test_cosine_similarity_returns_one_for_identical_vectors() -> None:
    """Identical non-zero vectors have a cosine similarity of one."""
    vector: NDArray[np.float64] = np.array([3.0, 4.0])

    assert retrieval.cosine_similarity(vector, vector) == 1.0


def test_retrieve_ranks_the_closest_document_first(monkeypatch: object) -> None:
    """Rank a matching document above an orthogonal document without an API call."""
    query_vector: NDArray[np.float64] = np.array([1.0, 0.0])
    monkeypatch.setattr(retrieval, "embed", lambda _: query_vector)

    results = retrieval.retrieve(
        "How do interest rates affect equities?",
        ["Interest rates affect equity valuations.", "AI spending supports growth."],
        [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        k=2,
    )

    assert results[0]["document"] == "Interest rates affect equity valuations."
    assert results[0]["score"] > results[1]["score"]
