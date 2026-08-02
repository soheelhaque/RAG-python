"""Offline tests for loading the text-file financial corpus."""

from pathlib import Path

from src.ingestion import load_documents


def test_load_documents_reads_text_files_in_filename_order(tmp_path: Path) -> None:
    """Load each non-empty text file in a deterministic order."""
    (tmp_path / "02_second.txt").write_text("Second document", encoding="utf-8")
    (tmp_path / "01_first.txt").write_text(" First document ", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("Ignored", encoding="utf-8")

    assert load_documents(tmp_path) == ["First document", "Second document"]
