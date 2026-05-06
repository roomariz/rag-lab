# RAG Benchmark Lab

A comprehensive toolkit for experimenting with and evaluating Retrieval-Augmented Generation (RAG) systems. Built on top of [Qdrant](https://qdrant.tech/) for vector storage and [Ollama](https://github.com/ollama/ollama) for local LLM inference.

## Features

- **Document Ingestion** - Chunk corpora into semantic segments and index into a vector store
- **Retrieval Pipeline** - Query vector stores with configurable top-k and score thresholds
- **RAGAs Evaluation** - Evaluate RAG systems using faithfulness, answer relevancy, context precision, and context recall
- **Benchmark Experiments**:
  - Top-K tuning
  - Chunking quality benchmarking
  - Labeled retrieval benchmarks
  - Hybrid retrieval
- **Streamlit Dashboard** - Interactive UI for all experimentation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run the Dashboard

```bash
streamlit run src/dashboard/app.py
```

The dashboard provides tabs for:
- **Retrieval Test** - Query your indexed corpus
- **Evaluation** - Run RAGAs evaluations
- **Embeddings** - Compare embedding models
- **Experiments** - Run benchmark experiments
- **Ingestion** - Index documents into Qdrant
- **Analytics** - View saved benchmark results

### 2. Programmatic Usage

```python
from src.ingestion import ingest_corpus
from src.retrieval import RetrievalPipeline

# Ingest a corpus
result = ingest_corpus(
    "path/to/docs",
    collection_name="my_collection",
    chunk_size=512,
    chunk_overlap=64,
)

# Run retrieval
pipeline = RetrievalPipeline(collection_name="my_collection", top_k=5)
result = pipeline.run("What is Docker used for?")
print(result.retrieved_chunks)
```

### 3. Run Benchmarks

```python
from src.benchmarks import (
    RetrievalExperiment,
    RetrievalBenchmarkDataset,
    run_chunking_quality_benchmark,
)

# Labeled retrieval benchmark
exp = RetrievalExperiment(collection_name="my_collection")
results = exp.run_labeled_retrieval_benchmark(dataset, k_values=[1, 3, 5])

# Chunking quality benchmark
result = run_chunking_quality_benchmark(
    documents=docs,
    dataset=dataset,
    strategies=strategies,
    top_k=5,
)
```

## Project Structure

```
src/
├── benchmarks/         # Benchmark experiments and evaluation
│   ├── chunking_experiments.py
│   ├── datasets.py
│   ├── evaluator.py
│   ├── experiments.py
│   ├── retrieval_metrics.py
│   └── artifacts.py
├── dashboard/         # Streamlit dashboard
│   └── app.py
├── ingestion/        # Document chunking and ingestion
│   ├── chunker.py
│   ├── ingest.py
│   └── manifest.py
├── retrieval/        # Retrieval pipeline
│   ├── pipeline.py
│   └── vector_store.py
├── visualization/     # Charts and analytics
│   ├── plots.py
│   └── benchmark_charts.py
└── config.py        # Configuration
```

## Configuration

Environment variables and settings can be configured in `src/config.py`:

- `llm_model` - LLM model for generation (default: llama3)
- `embed_model` - Embedding model (default: bge-m3)
- `chunk_size` - Default chunk size
- `chunk_overlap` - Default chunk overlap
- `qdrant_url` - Qdrant server URL
- `qdrant_api_key` - Qdrant API key

## Testing

```bash
pytest tests/
```

## Dependencies

- **ragas** - RAG evaluation framework
- **openai** - OpenAI-compatible API client
- **qdrant-client** - Vector database client
- **pandas** - Data processing
- **streamlit** - Dashboard UI
- **plotly** - Visualization (optional)