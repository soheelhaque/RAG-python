def build_prompt(query, retrieved_docs):
    context = "\n\n".join(
        [
            f"""Document:
{r['document']}

Relevance score: {r['score']:.3f}
Retrieval explanation: {r['explanation']}
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
