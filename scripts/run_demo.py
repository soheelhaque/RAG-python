from time import time

from src.rag_pipeline import rag


def main() -> None:
    """Run the sample financial question through the RAG pipeline.

    Returns:
        None: This function prints the generated answer and latency.
    """

    query = (
        "What are the key risks for US tech equities "
        "given interest rates and AI growth trends?"
    )

    start = time()

    result = rag(query)

    end = time()

    print("\n=== FINAL ANSWER ===\n")
    print(result["answer"])

    print(
        f"\nTotal latency: "
        f"{end - start:.2f} seconds"
    )

if __name__ == "__main__":
    main()
