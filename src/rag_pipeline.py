from data.financial_docs import documents
from src.embeddings import embed
from src.llm_client import call_llm
from src.prompt_builder import build_prompt
from src.retrieval import retrieve

doc_embeddings = [embed(doc) for doc in documents]


def rag(query):
    retrieved = retrieve(query, documents, doc_embeddings, k=3)

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

    return {
        "query": query,
        "retrieved_docs": retrieved,
        "prompt": prompt,
        "answer": answer
    }
