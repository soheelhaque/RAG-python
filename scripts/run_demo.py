from time import time

from src.rag_pipeline import rag

SAMPLE_QUESTION = (
    '\"What are the key risks for US tech equities given interest rates and AI growth trends?\"'
)


def get_question() -> str:
    """Prompt for a research question and fall back to the sample question."""
    question = input(
        "Enter a financial research question, or press Enter to use the sample question:\n"
        f"{SAMPLE_QUESTION}\n\n"
        "Question: "
    ).strip()
    return question or SAMPLE_QUESTION


def main() -> None:
    """Run a selected financial question through the RAG pipeline.

    Returns:
        None: This function prints the generated answer and latency.
    """

    query = get_question()

    start = time()

    result = rag(query)

    end = time()

    print("\n=== FINAL ANSWER ===\n")
    print(result["answer"])

    print(f"\nTotal latency: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
