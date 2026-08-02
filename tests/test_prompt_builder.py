"""Offline tests for prompt construction."""

from src.prompt_builder import build_prompt


def test_build_prompt_includes_question_and_retrieval_metadata() -> None:
    """Keep the query, source text, and score in the prompt."""
    prompt = build_prompt(
        "What are the risks?",
        [
            {
                "document": "Higher rates pressure growth stocks.",
                "score": 0.81234,
            }
        ],
    )

    assert "What are the risks?" in prompt
    assert "Higher rates pressure growth stocks." in prompt
    assert "Relevance score: 0.812" in prompt
