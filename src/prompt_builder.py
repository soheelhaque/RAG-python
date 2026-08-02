from collections.abc import Sequence

from src.types import RetrievedDocument


def build_prompt(query: str, retrieved_docs: Sequence[RetrievedDocument]) -> str:
    """Build the language-model prompt from a query and retrieved documents.

    Args:
        query (str): The user's financial research question.
        retrieved_docs (Sequence[RetrievedDocument]): Ranked documents and
            their retrieval metadata.

    Returns:
        str: A formatted prompt containing the context and response
            instructions.
    """

    context = "\n\n".join(
        [
            f"""Document:
{r['document']}

Relevance score: {r['score']:.3f}
"""
            for r in retrieved_docs
        ]
    )

    return f"""
You are a financial research assistant.

Use the retrieved context below. Pay attention to relevance scores.

Context:
{context}

Question:
{query}

Return:
- 5–10 bullet point summary
- optional risk notes
"""
