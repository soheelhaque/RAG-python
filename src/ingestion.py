"""Load the financial document corpus from text files."""

from pathlib import Path

FINANCIAL_DOCS_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "financial_docs"


def load_documents(directory: Path = FINANCIAL_DOCS_DIRECTORY) -> list[str]:
    """Return the non-empty text documents in ``directory`` in filename order."""
    documents = []
    for path in sorted(directory.glob("*.txt")):
        document = path.read_text(encoding="utf-8").strip()
        if document:
            documents.append(document)

    return documents
