"""Offline tests for interactive demo question selection."""

from pytest import MonkeyPatch

from scripts.run_demo import SAMPLE_QUESTION, get_documents, get_question


def test_get_question_uses_custom_input(monkeypatch: MonkeyPatch) -> None:
    """Return the user's question when they enter one."""
    monkeypatch.setattr("builtins.input", lambda _: "What are the earnings risks?")

    assert get_question() == "What are the earnings risks?"


def test_get_question_uses_sample_for_empty_input(monkeypatch: MonkeyPatch) -> None:
    """Return the sample question when the user presses Enter."""
    monkeypatch.setattr("builtins.input", lambda _: "")

    assert get_question() == SAMPLE_QUESTION


def test_get_documents_returns_short_corpus_for_first_option(monkeypatch: MonkeyPatch) -> None:
    """Select the in-code short corpus with option one."""
    monkeypatch.setattr("builtins.input", lambda _: "1")

    assert get_documents()[0].strip().startswith("US Equity Markets Outlook")


def test_get_documents_retries_invalid_selection(monkeypatch: MonkeyPatch, capsys: object) -> None:
    """Keep prompting until a valid numbered option is entered."""
    responses = iter(["3", "two", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    get_documents()

    assert capsys.readouterr().out.count("Invalid selection. Enter 1 or 2.") == 2
