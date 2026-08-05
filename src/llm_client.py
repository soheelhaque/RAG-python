import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LLM_MODEL = "gpt-4o-mini"

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def call_llm(prompt: str) -> str | None:
    """Generate an answer from the language model for the supplied prompt.

    Args:
        prompt (str): The complete instruction and context sent to the model.

    Returns:
        str | None: The model's response text, or ``None`` when the response
            has no text.
    """

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
