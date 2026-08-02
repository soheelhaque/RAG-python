from time import time

from data.financial_docs_short import documents as short_documents
from src.ingestion import load_documents
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


def get_documents() -> list[str]:
    """Prompt until the user chooses one of the available document corpora."""
    while True:
        selection = input(
            "Choose a document source:\n"
            "1. financial_docs_short.py\n"
            "2. financial_docs directory\n\n"
            "Selection: "
        ).strip()

        if selection == "1":
            return short_documents
        if selection == "2":
            return load_documents()

        print("Invalid selection. Enter 1 or 2.")


def main() -> None:
    """Run a selected financial question through the RAG pipeline.

    Returns:
        None: This function prints the generated answer and latency.
    """

    documents = get_documents()
    query = get_question()

    start = time()

    result = rag(query, documents)

    end = time()

    print("\n=== FINAL ANSWER ===\n")
    print(result["answer"])

    print(f"\nTotal latency: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
