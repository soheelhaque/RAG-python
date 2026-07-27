# RAG Framework Familiarisation Sprint
## Technical / Functional Specification

## 1. Purpose

This project is a small set of independent RAG implementations designed to build practical intuition across major LLM frameworks.

The objective is **not** to build a production RAG system.

The objective is to compare:

- framework abstractions
- developer experience
- implementation complexity
- debugging experience
- retrieval architecture choices
- mental models

Each implementation should use the same core business use case:

> Financial Research Assistant

Given a financial research question, retrieve relevant information from a small corpus and generate an investment-style summary.

---

# 2. Learning Objectives

By completing the four implementations, the developer should understand:

## Raw Python

How RAG works internally:

- embeddings
- similarity search
- retrieval
- prompt construction
- LLM generation

## LangChain

How mainstream RAG systems are structured:

- document loading
- text splitting
- vector stores
- retrievers
- chains / LCEL

## LlamaIndex

How document-centric RAG systems work:

- documents
- nodes
- indexing
- hierarchical retrieval
- hybrid retrieval

## Haystack

How pipeline-oriented systems work:

- explicit components
- retrieval stages
- reranking
- generation pipelines

---

# 3. Technology Stack

## Runtime

- Python 3.13.9

## Package Management

Use:

- uv

Requirements:

- use `pyproject.toml`
- commit `uv.lock`
- avoid pip requirements files

## Code Quality

Use:

- ruff

For:

- linting
- formatting
- import sorting

Configuration:

- line length: 100 characters
- standard Ruff rules:
  - E
  - F
  - I

## IDE Compatibility

The project must work with:

- VS Code
- JetBrains IDEs (PyCharm / IntelliJ Python plugin)

---

# 4. Project Structure

Each implementation must be an independent project.

Do NOT create one evolving codebase.

Projects:

```text
RAG-python/
RAG-langchain/
RAG-llamaindex/
RAG-haystack/
```

Each project should follow:

```text
project/
│
├── README.md
├── AGENTS.md
├── pyproject.toml
├── uv.lock
│
├── docs/
│   └── SPEC.md
│
├── data/
│   └── financial_docs/
│
├── src/
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── prompts.py
│   └── pipeline.py
│
└── scripts/
    └── run_demo.py
```

## Documentation

The `docs` directory contains project documentation.

The specification for each project must be stored as:

```text
docs/SPEC.md
```

---

# 5. AGENTS.md Requirements

Each project must contain an `AGENTS.md` file.

Purpose:

Provide persistent instructions for AI coding assistants such as Codex.

The file should contain:

- project purpose
- scope boundaries
- coding conventions
- technology constraints
- instructions not to introduce unnecessary complexity

Example principles:

- Keep implementations small and focused.
- Do not add production features unless explicitly requested.
- Prioritise learning value over abstraction.
- Prefer simple, readable code over clever solutions.
- Follow the architecture described in `docs/SPEC.md`.
- Use existing dependencies before introducing new ones.

---

# 6. Common Functional Requirements

All projects implement the same user journey.

## Input

A financial research question.

Example:

"What are the key risks for US technology equities given interest rates and AI growth trends?"

## Processing

The system must:

1. Retrieve relevant information.
2. Generate a concise investment-style response.

## Output

The response should contain:

- 5–10 bullet point summary
- optional risk notes section

---

# 7. Common Dataset

Use a small synthetic financial corpus.

Requirements:

- 5–10 documents
- stored as text files
- topics include:
  - equity markets
  - interest rates
  - AI investment trends
  - earnings
  - macroeconomic factors

The corpus should be identical across all frameworks.

Do not change the dataset between implementations.

---

# 8. Implementation 1 — RAG-python Baseline

## Purpose

Understand the mechanics of RAG without framework abstractions.

## Constraints

Do NOT use:

- LangChain
- LlamaIndex
- Haystack
- vector databases

## Implementation

Build manually:

### Embeddings

Convert documents and queries into vectors.

### Retrieval

Implement:

- cosine similarity
- ranking
- top-k selection

### Prompt Construction

Manually construct:

```text
Use the following context...

Context:
{retrieved documents}

Question:
{user question}
```

### Generation

Call the LLM API directly.

## Additional Requirement

Print retrieval debugging information:

- retrieved documents
- similarity scores
- ranking order

---

# 9. Implementation 2 — RAG-langchain Baseline

## Purpose

Understand standard RAG abstractions.

## Framework

Use:

- LangChain
- FAISS

## Requirements

Implement:

```text
Documents
    ↓
Text Splitter
    ↓
Embeddings
    ↓
FAISS Vector Store
    ↓
Retriever
    ↓
Prompt Template
    ↓
LLM
```

## Chunking

Use:

- fixed length chunking
- overlap between chunks

Example:

- chunk size: approximately 500 characters/tokens
- overlap: approximately 100 characters/tokens

Exact values are not important.

The purpose is understanding chunking.

## Retrieval

Use:

- simple semantic similarity retrieval

Do not add:

- reranking
- hybrid retrieval
- advanced query strategies

---

# 10. Implementation 3 — RAG-llamaindex Realistic RAG

## Purpose

Understand a more advanced document-oriented RAG architecture.

## Framework

Use:

- LlamaIndex
- FAISS
- BM25

## Requirements

Implement:

```text
Documents
    ↓
Hierarchical Chunking
    ↓
Dense Vector Index (FAISS)
             +
Sparse Retrieval (BM25)
    ↓
Hybrid Retrieval
    ↓
LLM Generation
```

## Chunking

Use hierarchical chunking:

Example:

- larger parent chunks
- smaller child chunks

The purpose is to understand:

- document structure
- node relationships
- retrieval granularity

## Retrieval

Implement:

- dense retrieval
- BM25 keyword retrieval
- combination of results

Do not add:

- agents
- memory
- evaluation frameworks

---

# 11. Implementation 4 — RAG-haystack RAG with Reranking

## Purpose

Understand explicit pipeline architectures.

## Framework

Use:

- Haystack
- InMemoryDocumentStore

Do not force FAISS integration.

## Requirements

Implement:

```text
Documents
    ↓
Document Store
    ↓
Embedding Retriever
    ↓
Top-N Candidates
    ↓
Reranker
    ↓
Prompt Builder
    ↓
LLM
```

## Chunking

Use:

- fixed length chunking
- overlap

## Retrieval

Implement:

1. Dense retrieval
2. Candidate selection
3. Reranking

The reranker should improve relevance before generation.

---

# 12. What NOT to Build

Do not add:

- user interface
- API layer
- authentication
- production deployment
- monitoring
- evaluation framework
- agents
- memory
- complex configuration systems

These belong in later projects.

---

# 13. Codex Usage Guidelines

Codex should accelerate implementation but not replace understanding.

Good uses:

- project scaffolding
- dependency setup
- boilerplate generation
- framework syntax assistance
- debugging errors
- explaining unfamiliar APIs

Avoid:

- generating the entire project without review
- adding unnecessary complexity
- introducing patterns not required by this specification

Preferred workflow:

1. Understand the architecture.
2. Ask Codex for implementation assistance.
3. Review generated code.
4. Run the project.
5. Modify and experiment.

---

# 14. Completion Criteria

Each project is complete when:

- it runs successfully
- it answers the sample query
- retrieval behaviour can be inspected
- README explains:
  - architecture
  - key framework concepts
  - what was easier/harder compared with previous implementations

The goal is learning, not production readiness.