# Scoped Task List

## 1. Native Qdrant implementation
- [x] Replace the Chroma-style code path with native Qdrant collection create/upsert/search/delete semantics.
- [x] Add proper payload metadata for source file, chunk id, offsets, chunking config, embedding model, and run id.
- [x] Make collection creation deterministic and configurable.
- [x] Add the Qdrant client dependency explicitly.
- [x] Optional tests: [tests/unit/test_vector_store.py](/mnt/d/workspace/open-source/ragas/tests/unit/test_vector_store.py) or equivalent.

### Target files
- [src/retrieval/vector_store.py](/mnt/d/workspace/open-source/ragas/src/retrieval/vector_store.py)
- [src/config.py](/mnt/d/workspace/open-source/ragas/src/config.py)
- [pyproject.toml](/mnt/d/workspace/open-source/ragas/pyproject.toml)
- [requirements.txt](/mnt/d/workspace/open-source/ragas/requirements.txt)

## 2. Reproducible ingestion and indexing
- [x] Add a corpus ingestion entry point for files/directories.
- [x] Persist a run manifest with corpus path, chunk config, embedding model, and collection name.
- [x] Make chunk metadata traceable back to the source text.
- [x] Ensure chunking config is configurable from a single place.
- [x] Optional tests:
  - [tests/unit/test_chunker.py](/mnt/d/workspace/open-source/ragas/tests/unit/test_chunker.py)
  - [tests/unit/test_ingest_manifest.py](/mnt/d/workspace/open-source/ragas/tests/unit/test_ingest_manifest.py)

### Target files
- [src/ingestion/chunker.py](/mnt/d/workspace/open-source/ragas/src/ingestion/chunker.py)
- [src/ingestion/__init__.py](/mnt/d/workspace/open-source/ragas/src/ingestion/__init__.py)
- [src/config.py](/mnt/d/workspace/open-source/ragas/src/config.py)

### New files
- [src/ingestion/ingest.py](/mnt/d/workspace/open-source/ragas/src/ingestion/ingest.py)
- [src/ingestion/manifest.py](/mnt/d/workspace/open-source/ragas/src/ingestion/manifest.py)

## 3. Proper retrieval benchmarks
- [x] Add retrieval metrics such as recall@k, hit rate, and query-to-chunk match scoring.
- [x] Keep retrieval latency separate from generation latency.
- [x] Save per-query benchmark results in structured form.
- [x] Support labeled query sets with expected relevant chunks or documents.
- [x] Optional tests: [tests/unit/test_retrieval_metrics.py](/mnt/d/workspace/open-source/ragas/tests/unit/test_retrieval_metrics.py)

### Target files
- [src/benchmarks/evaluator.py](/mnt/d/workspace/open-source/ragas/src/benchmarks/evaluator.py)
- [src/benchmarks/experiments.py](/mnt/d/workspace/open-source/ragas/src/benchmarks/experiments.py)
- [src/retrieval/pipeline.py](/mnt/d/workspace/open-source/ragas/src/retrieval/pipeline.py)

### New files
- [src/benchmarks/retrieval_metrics.py](/mnt/d/workspace/open-source/ragas/src/benchmarks/retrieval_metrics.py)
- [src/benchmarks/datasets.py](/mnt/d/workspace/open-source/ragas/src/benchmarks/datasets.py)

## 4. Chunking-quality experiments
- [x] Extend chunk strategy benchmarking beyond chunk count and average size.
- [x] Compare retrieval quality across chunk size, overlap, and separator variants.
- [x] Emit results that can be ranked and charted.
- [x] Dedicated chunking benchmark module is now implemented and wired into the dashboard.
- [x] Optional tests: [tests/unit/test_chunking_experiments.py](/mnt/d/workspace/open-source/ragas/tests/unit/test_chunking_experiments.py)

### Target files
- [src/ingestion/chunker.py](/mnt/d/workspace/open-source/ragas/src/ingestion/chunker.py)
- [src/benchmarks/experiments.py](/mnt/d/workspace/open-source/ragas/src/benchmarks/experiments.py)

### New files
- [src/benchmarks/chunking_experiments.py](/mnt/d/workspace/open-source/ragas/src/benchmarks/chunking_experiments.py)

## 5. Embedding comparison
- [x] Compare `nomic-embed-text` against other local embedding models on quality and latency.
- [x] Add consistent benchmark outputs for faithfulness, recall, and retrieval behavior.
- [x] Make model comparison reusable by the dashboard.
- [x] The dashboard can compare embedding latency, retrieval quality, and faithfulness across models and save the artifact.
- [x] Optional tests: [tests/unit/test_embedding_benchmarks.py](/mnt/d/workspace/open-source/ragas/tests/unit/test_embedding_benchmarks.py)

### Target files
- [src/benchmarks/evaluator.py](/mnt/d/workspace/open-source/ragas/src/benchmarks/evaluator.py)
- [src/benchmarks/experiments.py](/mnt/d/workspace/open-source/ragas/src/benchmarks/experiments.py)
- [src/dashboard/app.py](/mnt/d/workspace/open-source/ragas/src/dashboard/app.py)

### New files
- [src/benchmarks/embedding_benchmarks.py](/mnt/d/workspace/open-source/ragas/src/benchmarks/embedding_benchmarks.py)

## 6. Latency instrumentation
- [x] Standardize duration fields for chunking, embedding, indexing, retrieval, generation, and evaluation.
- [x] Use one timing schema across all outputs.
- [x] Make latency data available to the dashboard and plots without reshaping.
- [x] The dashboard and plots now consume the same timing schema, including ingestion and chunking timing.
- [x] Optional tests: [tests/unit/test_timing.py](/mnt/d/workspace/open-source/ragas/tests/unit/test_timing.py)

### Target files
- [src/retrieval/pipeline.py](/mnt/d/workspace/open-source/ragas/src/retrieval/pipeline.py)
- [src/retrieval/vector_store.py](/mnt/d/workspace/open-source/ragas/src/retrieval/vector_store.py)
- [src/ingestion/chunker.py](/mnt/d/workspace/open-source/ragas/src/ingestion/chunker.py)
- [src/benchmarks/evaluator.py](/mnt/d/workspace/open-source/ragas/src/benchmarks/evaluator.py)
- [src/benchmarks/experiments.py](/mnt/d/workspace/open-source/ragas/src/benchmarks/experiments.py)

### New files
- [src/benchmarks/timing.py](/mnt/d/workspace/open-source/ragas/src/benchmarks/timing.py)

## 7. Dashboard integration
- [x] Wire the UI to real ingestion, indexing, retrieval, and benchmark outputs.
- [x] Add controls for corpus ingestion, chunk strategy, top-k, and embedding model.
- [x] Show retrieved chunks, generated answer, metrics, and latency from the same run.
- [x] Add saved-run browsing instead of only one-off demos.

### Target files
- [src/dashboard/app.py](/mnt/d/workspace/open-source/ragas/src/dashboard/app.py)
- [src/visualization/plots.py](/mnt/d/workspace/open-source/ragas/src/visualization/plots.py)

### New files
- [src/dashboard/components.py](/mnt/d/workspace/open-source/ragas/src/dashboard/components.py)
- [src/dashboard/state.py](/mnt/d/workspace/open-source/ragas/src/dashboard/state.py)

## 8. Visual analytics
- [ ] Optional tests: Streamlit smoke test or simple import-level test if the repo has that pattern.
- [ ] Add t-SNE cluster views fed by persisted embeddings.
- [ ] Add retrieval neighborhood inspection for individual queries.
- [ ] Add trend charts for metrics and latency over time.
- [x] Make the plots consume experiment artifacts directly.
- [ ] Optional tests: [tests/unit/test_plots.py](/mnt/d/workspace/open-source/ragas/tests/unit/test_plots.py)

### Target files
- [src/visualization/plots.py](/mnt/d/workspace/open-source/ragas/src/visualization/plots.py)
- [src/dashboard/app.py](/mnt/d/workspace/open-source/ragas/src/dashboard/app.py)

### New files
- [src/visualization/embedding_views.py](/mnt/d/workspace/open-source/ragas/src/visualization/embedding_views.py)
- [src/visualization/trends.py](/mnt/d/workspace/open-source/ragas/src/visualization/trends.py)

## Recommended dependency order
- [ ] Start with steps 1 and 2.
- [ ] Then do step 3 and step 4 together, because they share the same corpus/indexing foundation.
- [ ] Do step 5 and step 6 next.
- [x] Finish with step 7 and step 8.
