from time import time

from data.financial_docs_short import documents as short_documents
from src.ingestion import load_documents
from src.rag_pipeline import rag

SAMPLE_QUESTION = (
    "What are the key risks for US tech equities given interest rates and AI growth trends?"
)
FINANCIAL_DOCS_QUESTION = (
    "How can delayed power connections and slower-than-expected AI workload utilisation affect"
    " a cloud provider's cash flow, depreciation, and valuation?"
)


def get_question(document_source: str) -> str:
    """Prompt for a research question with a source-specific sample fallback."""
    sample_question = (
        SAMPLE_QUESTION if document_source == "1" else FINANCIAL_DOCS_QUESTION
    )
    question = input(
        "Enter a financial research question, or press Enter to use the sample question:\n"
        f"{sample_question}\n\n"
        "Question: "
    ).strip()
    return question or sample_question


def get_documents() -> tuple[list[str], str]:
    """Prompt until the user chooses a document corpus and return its source."""
    while True:
        selection = input(
            "Choose a document source:\n"
            "1. financial_docs_short.py\n"
            "2. financial_docs directory\n\n"
            "Selection: "
        ).strip()

        if selection == "1":
            return short_documents, selection
        if selection == "2":
            return load_documents(), selection

        print("Invalid selection. Enter 1 or 2.")


def main() -> None:
    """Run a selected financial question through the RAG pipeline.

    Returns:
        None: This function prints the generated answer and latency.
    """

    documents, document_source = get_documents()
    query = get_question(document_source)

    start = time()

    result = rag(query, documents)

    end = time()

    print("\n=== FINAL ANSWER ===\n")
    print(result["answer"])

    print(f"\nTotal latency: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
