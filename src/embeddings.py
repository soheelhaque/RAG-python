import os

import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def embed(text: str) -> NDArray[np.float64]:
    """Create a numeric embedding for a piece of text using OpenAI.

    Args:
        text (str): The text to convert into an embedding vector.

    Returns:
        NDArray[np.float64]: A NumPy array containing the text embedding as
            64-bit floats.
    """

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float64)
