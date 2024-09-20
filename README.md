# WikiQA: A RAG Pipeline Powered by Google Search and Wikipedia

## Overview

WikiQA is a Retrieval-Augmented Generation (RAG) pipeline built for answering questions by retrieving the most relevant wikipedia articles using the Wikipedia API and the Google Search API.

---

## Libraries:

- **Gemini**: Used for generating an eval dataset and as a judge.
- **LlamaIndex**: Core framework for building the RAG pipeline.
- **HuggingFace**: Open-Source `Phi-3.5` SLM and `bge-base-en` embeddings.
- **ChromaDB**: Open-source persistent vector DB.
- **Ragas**: Evaluating the RAG pipeline.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ahmedo42/wikiqa.git
    cd wikiqa
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the pipeline demo:
    ```bash
    jupyter notebook demo.ipynb
    ```

## File Structure

- `rag.py`: Core RAG pipeline implementation.
- `helpers.py`: Utility functions for data preprocessing and query handling.
- `build_eval_dataset.ipynb`: Notebook to construct a dataset for testing using Google's Gemini.
- `demo.ipynb`: Interactive demo of the pipeline.
- `eval.ipynb`: Evaluate the RAG pipeline using the generated eval dataset.