"""Shared type definitions for the retrieval-augmented generation pipeline."""

from typing import TypedDict


class RetrievedDocument(TypedDict):
    """A document returned by the retrieval step with ranking metadata."""

    score: float
    document: str


class RAGResponse(TypedDict):
    """The complete set of inputs, context, and output from a RAG query."""

    query: str
    retrieved_docs: list[RetrievedDocument]
    prompt: str
    answer: str | None
