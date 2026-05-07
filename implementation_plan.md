# Implementation Backlog

## 1. Native Qdrant implementation
- [x] Qdrant-backed `VectorStore` with real `upsert`, `query`, delete, and collection info methods.
- [x] Metadata persistence for source document, chunk id, offsets, chunk strategy, embed model, and run id.
- [x] Top-k similarity retrieval with configurable score threshold.
- [x] Works without Chroma fallback in the main path.
- [x] Ingested chunks can be queried back with scores and metadata.
- [x] Collection creation is deterministic and repeatable.
- [x] This is the current blocker for everything else.

## 2. Reproducible ingestion/indexing
- [x] A document ingestion command or service that loads files, chunks them, embeds them, and indexes them into Qdrant.
- [x] Configurable chunk size, overlap, separator strategy, and embedding model.
- [x] Dataset/run manifests saved to disk for reproducibility.
- [x] Same input corpus and config produce the same indexed artifact set.
- [x] Indexing can be rerun from a manifest without manual steps.
- [x] Chunk metadata is sufficient to trace every retrieved chunk back to source text.
- [x] This should become the canonical entry point for building test corpora.

## 3. Proper retrieval benchmarks
- [x] Benchmark harness for retrieval accuracy, top-k hit rate, recall@k, and latency.
- [x] Query set support with expected relevant documents/chunks.
- [x] Structured result output as CSV or JSONL.
- [x] Benchmarks report per-query and aggregate metrics.
- [x] Results are comparable across runs and parameter sets.
- [x] Retrieval latency is measured separately from generation latency.
- [x] This should be built against the ingestion/indexing pipeline, not synthetic in-memory data.

## 4. Chunking-quality experiments
- [x] Experiment runner for chunk size, overlap, and separator variants.
- [x] Evaluation of downstream retrieval quality for each strategy.
- [x] Summary tables and charts for strategy comparison.
- [x] Can compare at least several chunking configs on the same corpus and query set.
- [x] Reports both structural stats and retrieval quality metrics.
- [x] Produces reproducible experiment artifacts.
- [x] Focus on quality impact, not just chunk counts.
- [x] Current benchmark layer and dashboard view now cover chunking-quality experiments end to end.

## 5. Embedding comparison
- [x] Comparison harness for `nomic-embed-text` and other local embedding models.
- [x] Metrics for retrieval quality, faithfulness, and embedding latency.
- [ ] Optionally embedding cost/resource usage if available locally.
- [x] Same corpus/query set can be run across multiple models.
- [x] Results include quality plus latency, not latency only.
- [x] Output is normalized enough to compare models directly.
- [x] Treat this as a benchmark layer, not a UI feature.
- [x] Current dashboard coverage now spans latency, retrieval quality, and faithfulness comparisons across embedding models.

## 6. Latency instrumentation everywhere
- [x] Timing for ingestion, chunking, embedding, indexing, retrieval, reranking if added, and generation.
- [x] Unified result schema that records timestamps and durations.
- [x] Latency breakdown views for per-run and aggregate reporting.
- [x] Every experiment output includes comparable timing fields.
- [x] Latency is measured in a consistent way across benchmarks.
- [x] Dashboard and plots can consume the same timing schema.
- [x] This should be threaded through all earlier steps, not added after the fact.
- [x] Current coverage now includes ingestion and chunking timing normalized into the same schema.

## 7. Dashboard integration
- [x] Streamlit app wired to real ingestion, indexing, retrieval, and benchmark outputs.
- [x] Views for retrieved chunks, generated answers, metrics, and timing.
- [x] Controls for chunking config, top-k, embedding model, and experiment selection.
- [x] A user can run the end-to-end pipeline from the UI.
- [x] The dashboard shows actual persisted experiment results.
- [x] It exposes enough configuration to reproduce a run.
- [ ] Keep it operational, not decorative.

## 8. Visual analytics
- [ ] t-SNE or similar embedding cluster views.
- [ ] Retrieval neighborhood inspection.
- [ ] Evaluation trend charts over time.
- [ ] Score/latency distribution plots.
- [x] Visualizations are fed from saved experiment artifacts.
- [ ] Plots help diagnose retrieval behavior, not just summarize it.
- [x] Dashboard can render the key views without manual data shaping.
- [ ] These should sit on top of the experiment schema, not drive it.

## Recommended execution rule
- [ ] Build each layer so it writes a durable artifact: manifest, run config, metrics table, and raw outputs.
- [ ] Treat that artifact schema as the contract between retrieval, benchmarking, dashboard, and analytics.
- [ ] Don’t start the dashboard until the artifact schema is stable.

If you want, I can turn this into a week-by-week build plan or a scoped task list with file-level implementation targets.
