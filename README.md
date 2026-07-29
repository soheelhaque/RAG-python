# RAG-python

## Project overview

This repository contains a small, educational retrieval-augmented generation (RAG) implementation for a financial research assistant. The goal is to demonstrate the core mechanics of a simple RAG workflow in Python rather than build a production-ready system.

The project shows how the application:

- loads a small synthetic financial corpus
- creates embeddings for the documents
- retrieves the most relevant passages for a user query
- builds a prompt and generates an investment-style summary with an LLM

This is intended as a baseline for learning and comparison with other RAG frameworks.

## RAG sequence

The demo follows this retrieval-augmented generation flow:

```mermaid
sequenceDiagram
    participant Demo as Demo Script
    participant Pipeline as RAG Pipeline
    participant Retrieval as Retrieval Module
    participant Embeddings as Embeddings Module
    participant Prompt as Prompt Builder
    participant LLMClient as LLM Client
    participant OpenAI as OpenAI API

    Demo->>Pipeline: rag(query)
    Pipeline->>Retrieval: retrieve(query, documents, doc_embeddings, k=3)
    Retrieval->>Embeddings: embed(query)
    Embeddings->>OpenAI: Create query embedding
    OpenAI-->>Embeddings: Return query vector
    Embeddings-->>Retrieval: Return query vector
    Retrieval->>Retrieval: Calculate similarity and rank documents
    Retrieval-->>Pipeline: Return top 3 retrieved documents
    Pipeline->>Prompt: build_prompt(query, retrieved)
    Prompt-->>Pipeline: Return prompt
    Pipeline->>LLMClient: call_llm(prompt)
    LLMClient->>OpenAI: Create chat completion
    OpenAI-->>LLMClient: Return generated answer
    LLMClient-->>Pipeline: Return answer
    Pipeline-->>Demo: Return RAG response
    Demo->>Demo: Print answer and latency
```

## Installation

This project uses Python 3.13+ and the package manager uv.

1. Install uv if it is not already available.
2. From the repository root, install dependencies:

   ```bash
   uv sync
   ```

3. Create a `.env` file in the project root and add your OpenAI API key:

   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```

## Running the demo

Run the sample RAG workflow from the project root:

```bash
uv run python scripts/run.py
```

This will execute a sample financial research query and print the generated answer along with the total runtime.
