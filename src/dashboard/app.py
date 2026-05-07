from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from uuid import uuid4

from src.config import config, RESULTS_DIR
from src.retrieval import RetrievalPipeline
from src.retrieval.vector_store import VectorStore
from src.retrieval.vector_store import QDRANT_AVAILABLE
from src.ingestion import DocumentChunker, ingest_corpus
from src.ingestion.ingest import preview_supported_sources
from src.benchmarks import (
    RAGEvaluator,
    RetrievalExperiment,
    RetrievalBenchmarkDataset,
    ChunkingStrategySpec,
    run_chunking_quality_benchmark,
    save_chunking_benchmark,
    run_embedding_comparison,
    save_embedding_comparison,
    BenchmarkArtifact,
    list_benchmark_artifacts,
    load_benchmark_artifact,
    save_benchmark_artifact,
)
from src.visualization import create_benchmark_analytics_charts

st.set_page_config(
    page_title="RAG Benchmark Lab",
    page_icon="",
    layout="wide",
)

st.title("RAG Benchmark Lab")

st.sidebar.header("Configuration")
st.sidebar.markdown(f"**LLM Model:** {config.llm_model}")
st.sidebar.markdown(f"**Embed Model:** {config.embed_model}")
st.sidebar.markdown(f"**Qdrant Client:** {'available' if QDRANT_AVAILABLE else 'missing'}")

if not QDRANT_AVAILABLE:
    st.warning(
        "qdrant-client is not installed in this Python environment. "
        "Retrieval, ingestion, and benchmark runs that need the vector store will be disabled until it is installed."
    )

def _save_uploaded_files(uploaded_files, target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for uploaded in uploaded_files:
        output_path = target_dir / uploaded.name
        output_path.write_bytes(uploaded.getbuffer())
        saved.append(output_path)
    return saved


def _serialize_summary(summary: dict) -> dict:
    serialized = {}
    for key, value in summary.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            serialized[key] = value
        else:
            serialized[key] = str(value)
    return serialized


def _save_dashboard_artifact(
    *,
    artifact_type: str,
    experiment_name: str,
    results: pd.DataFrame,
    summary: dict,
    config_payload: dict,
    metadata: dict | None = None,
) -> Path:
    artifact = BenchmarkArtifact(
        artifact_type=artifact_type,
        experiment_name=experiment_name,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        config=config_payload,
        summary=_serialize_summary(summary),
        results=results,
        metadata=metadata or {},
    )
    return save_benchmark_artifact(artifact)


def _artifact_label(path: Path) -> str:
    try:
        artifact = load_benchmark_artifact(path)
    except Exception:
        return path.name

    created = artifact.timestamp[:19] if artifact.timestamp else "unknown time"
    return f"{artifact.experiment_name} · {artifact.artifact_type} · {created}"


def _parse_document_records(raw_json: str) -> list[dict]:
    parsed = json.loads(raw_json)
    if not isinstance(parsed, list):
        raise ValueError("Corpus documents must be a JSON list.")
    return parsed


def _parse_query_records(raw_json: str) -> list[dict]:
    parsed = json.loads(raw_json)
    if not isinstance(parsed, list):
        raise ValueError("Query records must be a JSON list.")
    return parsed


def _parse_strategy_records(raw_json: str) -> list[dict]:
    parsed = json.loads(raw_json)
    if not isinstance(parsed, list):
        raise ValueError("Chunking strategies must be a JSON list.")
    return parsed


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Retrieval Test",
    "Evaluation",
    "Embedding Comparison",
    "Experiments",
    "Ingestion",
    "Saved Analytics",
])

with tab1:
    st.header("Retrieval Pipeline Test")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Query Input")
        query = st.text_area("Enter your query:", value="What is Docker used for?")
        retrieval_collection = st.text_input(
            "Collection name",
            value=st.session_state.get("last_collection_name", "demo"),
            key="retrieval_collection_name",
        )

        top_k = st.slider("Top K", 1, 10, 5)
        score_threshold = st.slider("Score Threshold", 0.0, 1.0, 0.0, 0.1)

        run_btn = st.button("Run Retrieval", type="primary")

    with col2:
        if run_btn and query:
            if not QDRANT_AVAILABLE:
                st.error("Retrieval is unavailable because qdrant-client is missing in this environment.")
            else:
                pipeline = RetrievalPipeline(
                    collection_name=retrieval_collection,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )

                if not pipeline.vector_store.collection_exists():
                    st.error(
                        f"Collection '{retrieval_collection}' does not exist yet. "
                        "Run ingestion first or choose a collection that already has indexed chunks."
                    )
                else:
                    with st.spinner("Running retrieval..."):
                        result = pipeline.run(query, include_generation=True)

                    st.subheader("Results")

                    st.metric("Retrieval Latency", f"{result.retrieval_latency:.3f}s")
                    st.metric("Generation Latency", f"{result.generation_latency:.3f}s")
                    st.metric("Total Latency", f"{result.total_latency:.3f}s")

                    st.subheader("Retrieved Chunks")
                    for i, chunk in enumerate(result.retrieved_chunks):
                        with st.expander(f"Chunk {i+1} (score: {chunk.score:.3f})"):
                            st.text(chunk.text)
                            st.json(chunk.metadata)

                    if result.generated_response:
                        st.subheader("Generated Response")
                        st.text(result.generated_response)

with tab2:
    st.header("RAGAs Evaluation")

    if RAGEvaluator is None:
        st.warning("RAGAS evaluation dependencies are unavailable in this environment.")
    else:
        st.info("RAGAs evaluation requires the pipeline to return retrieved contexts, generated responses, and reference answers.")

        if st.button("Run Quick Evaluation"):
            evaluator = RAGEvaluator()

            test_data = [
                {
                    "query": "What is Docker used for?",
                    "contexts": ["Docker is an open platform for developing, shipping, and running applications using lightweight portable containers."],
                    "response": "Docker is used to develop and run applications inside portable containers.",
                    "reference": "Docker runs applications inside containers.",
                },
                {
                    "query": "What is Kubernetes?",
                    "contexts": ["Kubernetes is an open-source container orchestration platform for automating deployment and scaling."],
                    "response": "Kubernetes automates deployment and scaling of containerized applications.",
                    "reference": "Kubernetes is used for container orchestration.",
                },
            ]

            with st.spinner("Running evaluation..."):
                result = evaluator.evaluate(
                    queries=[d["query"] for d in test_data],
                    retrieved_contexts=[d["contexts"] for d in test_data],
                    generated_responses=[d["response"] for d in test_data],
                    references=[d["reference"] for d in test_data],
                )

            st.success("Evaluation complete!")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Faithfulness", f"{result.metrics.get('faithfulness', 0):.2f}")
            col2.metric("Answer Relevancy", f"{result.metrics.get('answer_relevancy', 0):.2f}")
            col3.metric("Context Precision", f"{result.metrics.get('context_precision', 0):.2f}")
            col4.metric("Context Recall", f"{result.metrics.get('context_recall', 0):.2f}")

            st.subheader("Per-Sample Results")
            st.dataframe(result.per_sample_results)

            artifact_path = _save_dashboard_artifact(
                artifact_type="ragas_evaluation",
                experiment_name=result.experiment_name,
                results=result.per_sample_results,
                summary=result.metrics,
                config_payload={
                    "llm_model": evaluator.llm_model,
                    "embed_model": evaluator.embed_model,
                },
                metadata={"latencies": result.latencies, "timings": result.timings, "timestamp": result.timestamp},
            )
            st.caption(f"Saved benchmark artifact: {artifact_path}")

with tab3:
    st.header("Embedding Model Comparison")

    if run_embedding_comparison is None:
        st.warning("Embedding comparison dependencies are unavailable in this environment.")
    else:
        st.info("Compare embedding latency, retrieval quality, and RAG faithfulness across local models.")

        default_corpus = [
            {"name": "docker.md", "text": "Docker packages applications into portable containers."},
            {"name": "kubernetes.md", "text": "Kubernetes orchestrates containerized workloads across clusters."},
            {"name": "security.md", "text": "SQL injection is a code injection attack that targets databases."},
        ]
        default_queries = [
            {
                "query": "What is Docker used for?",
                "relevant_sources": ["docker.md"],
                "reference": "Docker packages applications into portable containers.",
            },
            {
                "query": "What does Kubernetes do?",
                "relevant_sources": ["kubernetes.md"],
                "reference": "Kubernetes orchestrates containerized workloads across clusters.",
            },
        ]

        corpus_raw = st.text_area(
            "Corpus documents (JSON list):",
            value=json.dumps(default_corpus, indent=2),
            height=220,
        )

        query_raw = st.text_area(
            "Labeled queries (JSON list):",
            value=json.dumps(default_queries, indent=2),
            height=220,
        )

        models = st.multiselect(
            "Models to compare:",
            ["nomic-embed-text:latest", "mxbai-embed-large:latest", "bge-m3:latest"],
            default=["nomic-embed-text:latest"],
        )

        col1, col2 = st.columns(2)
        with col1:
            num_runs = st.slider("Embedding latency runs", 1, 10, 3)
        with col2:
            top_k = st.slider("Retrieval top-k", 1, 10, 5)

        if st.button("Run Comparison", type="primary") and models:
            try:
                corpus_records = _parse_document_records(corpus_raw)
                query_records = _parse_query_records(query_raw)
            except ValueError as exc:
                st.error(str(exc))
            else:
                with st.spinner("Running embedding comparison..."):
                    result = run_embedding_comparison(
                        texts=corpus_records,
                        models=models,
                        num_runs=num_runs,
                        query_records=query_records,
                        top_k=top_k,
                        llm_model=config.llm_model,
                    )

                st.subheader("Model Summary")
                st.dataframe(result.per_model_results)

                chart_frame = result.per_model_results.copy()
                numeric_columns = [
                    column
                    for column in [
                        "mean_latency",
                        "document_embedding_latency",
                        "query_embedding_latency",
                        "mean_recall_at_k",
                        "mean_faithfulness",
                        "mean_retrieval_latency",
                    ]
                    if column in chart_frame.columns
                ]
                if numeric_columns:
                    st.bar_chart(chart_frame.set_index("model")[numeric_columns])

                if not result.per_query_results.empty:
                    st.subheader("Per-Query Retrieval Results")
                    st.dataframe(result.per_query_results)

                if not result.per_sample_results.empty:
                    st.subheader("Per-Sample Faithfulness Results")
                    st.dataframe(result.per_sample_results)

                artifact_path = save_embedding_comparison(result)
                st.caption(f"Saved benchmark artifact: {artifact_path}")

with tab4:
    st.header("Retrieval Experiments")

    st.info("Run experiments to tune retrieval parameters.")

    experiment_type = st.selectbox(
        "Experiment Type",
        ["Top-K Tuning", "Chunking Quality Benchmark", "Labeled Retrieval Benchmark", "Hybrid Retrieval"],
    )

    if experiment_type == "Top-K Tuning":
        st.subheader("Top-K Tuning Experiment")

        queries = st.text_area(
            "Test queries (one per line):",
            value="What is Docker?\nWhat is Kubernetes?",
        ).strip().split("\n")

        k_values = st.multiselect(
            "K values to test:",
            [1, 2, 3, 5, 10, 15, 20],
            default=[1, 3, 5, 10],
        )

        if st.button("Run Experiment") and queries and k_values:
            from src.benchmarks import RetrievalExperiment

            if not QDRANT_AVAILABLE:
                st.error("Top-K tuning requires qdrant-client in this environment.")
            else:
                exp = RetrievalExperiment(collection_name="topk_tune")
                query_data = [{"question": q} for q in queries]

                with st.spinner("Running experiment..."):
                    results = exp.run_top_k_tuning(query_data, k_values)

                st.subheader("Results")
                st.dataframe(results)

                st.subheader("Latency by K")
                latency_by_k = results.groupby("top_k")[["retrieval_latency", "retrieval_duration"]].mean()
                st.line_chart(latency_by_k)

                artifact_path = _save_dashboard_artifact(
                    artifact_type="top_k_tuning",
                    experiment_name="top_k_tuning",
                    results=results,
                    summary={
                        "num_queries": len(queries),
                        "num_k_values": len(k_values),
                        "mean_retrieval_latency": float(results["retrieval_latency"].mean()) if not results.empty else 0.0,
                        "mean_retrieval_duration": float(results["retrieval_duration"].mean()) if not results.empty and "retrieval_duration" in results.columns else 0.0,
                    },
                    config_payload={
                        "collection_name": exp.collection_name,
                        "k_values": k_values,
                    },
                )
                st.caption(f"Saved benchmark artifact: {artifact_path}")

    elif experiment_type == "Chunking Quality Benchmark":
        st.subheader("Chunking Quality Benchmark")

        st.info(
            "Benchmark multiple chunking strategies against the same labeled query set. "
            "The benchmark saves a durable artifact and reports both chunk structure and retrieval quality."
        )

        corpus_json = st.text_area(
            "Corpus documents (JSON list with `name` and `text`):",
            value=json.dumps(
                [
                    {
                        "name": "docker.md",
                        "text": "Docker is a platform for packaging applications into lightweight containers.",
                    },
                    {
                        "name": "kubernetes.md",
                        "text": "Kubernetes orchestrates containers and automates deployment at scale.",
                    },
                    {
                        "name": "sql.md",
                        "text": "SQL injection is a vulnerability where attackers manipulate SQL queries.",
                    },
                ],
                indent=2,
            ),
            height=220,
        )

        chunking_query_json = st.text_area(
            "Labeled queries (JSON list):",
            value=json.dumps(
                [
                    {"query": "What is Docker used for?", "relevant_sources": ["docker.md"]},
                    {"query": "What is Kubernetes?", "relevant_sources": ["kubernetes.md"]},
                    {"query": "What is SQL injection?", "relevant_sources": ["sql.md"]},
                ],
                indent=2,
            ),
            height=220,
        )

        strategy_json = st.text_area(
            "Chunking strategies (JSON list):",
            value=json.dumps(
                [
                    {"name": "small_chunks", "chunk_size": 256, "chunk_overlap": 32},
                    {"name": "balanced_chunks", "chunk_size": 512, "chunk_overlap": 64},
                    {"name": "wide_chunks", "chunk_size": 768, "chunk_overlap": 96},
                ],
                indent=2,
            ),
            height=240,
        )

        chunking_collection_prefix = st.text_input(
            "Collection prefix:",
            value="chunking_quality",
        )
        chunking_top_k = st.slider("Top K for benchmark", 1, 10, 5)

        if st.button("Run Chunking Benchmark", type="primary"):
            try:
                corpus_documents = _parse_document_records(corpus_json)
                queries = _parse_strategy_records(chunking_query_json)
                strategies = [ChunkingStrategySpec.from_dict(item) for item in _parse_strategy_records(strategy_json)]
                dataset = RetrievalBenchmarkDataset.from_records(
                    queries,
                    name="chunking_quality_benchmark",
                    metadata={"source": "dashboard", "experiment": "chunking_quality"},
                )
            except Exception as exc:
                st.error(f"Invalid chunking benchmark input: {exc}")
            else:
                if not QDRANT_AVAILABLE:
                    st.error("Chunking quality benchmarking requires qdrant-client in this environment.")
                else:
                    with st.spinner("Running chunking benchmark..."):
                        result = run_chunking_quality_benchmark(
                            documents=corpus_documents,
                            dataset=dataset,
                            strategies=strategies,
                            top_k=chunking_top_k,
                            embed_model=config.embed_model,
                            collection_prefix=chunking_collection_prefix,
                            cleanup_collections=True,
                        )

                    st.success("Chunking benchmark complete.")
                    st.subheader("Strategy Summary")
                    st.dataframe(result.strategy_summary)

                    st.subheader("Per-Query Results")
                    st.dataframe(result.per_query_results)

                    if not result.strategy_summary.empty:
                        st.subheader("Retrieval Quality by Strategy")
                        summary_chart = result.strategy_summary.set_index("chunking_strategy")[
                            ["mean_recall_at_k", "mean_precision_at_k", "mean_hit_rate", "mean_mrr"]
                        ]
                        st.bar_chart(summary_chart)

                        st.subheader("Timing by Strategy")
                        timing_chart = result.strategy_summary.set_index("chunking_strategy")[
                            ["chunking_duration", "embedding_duration", "indexing_duration", "total_duration"]
                        ]
                        st.bar_chart(timing_chart)

                        st.subheader("Chunk Structure by Strategy")
                        st.dataframe(
                            result.strategy_summary[
                                [
                                    "chunking_strategy",
                                    "chunk_size",
                                    "chunk_overlap",
                                    "num_chunks",
                                    "avg_chunk_size",
                                    "mean_recall_at_k",
                                    "mean_precision_at_k",
                                    "mean_retrieval_duration",
                                ]
                            ]
                        )

                    artifact_path = save_chunking_benchmark(
                        result,
                        strategies=strategies,
                        documents=corpus_documents,
                    )
                    st.caption(f"Saved benchmark artifact: {artifact_path}")

    elif experiment_type == "Labeled Retrieval Benchmark":
        st.subheader("Labeled Retrieval Benchmark")

        st.info(
            "Provide labeled queries with relevant sources or exact chunk IDs. "
            "This benchmark reports recall@k, hit rate, precision@k, MRR, and latency."
        )

        benchmark_collection = st.text_input(
            "Collection name:",
            value=st.session_state.get("last_collection_name", "demo"),
        )
        benchmark_name = st.text_input("Dataset name:", value="local_retrieval_benchmark")
        benchmark_k_values = st.multiselect(
            "Top-K values:",
            [1, 2, 3, 5, 10, 15, 20],
            default=[1, 3, 5, 10],
        )
        benchmark_query_json = st.text_area(
            "Labeled queries (JSON list):",
            value=json.dumps(
                [
                    {
                        "query": "What is Docker used for?",
                        "relevant_sources": ["docker.md"],
                    },
                    {
                        "query": "What is Kubernetes?",
                        "relevant_sources": ["kubernetes.md"],
                    },
                ],
                indent=2,
            ),
            height=220,
        )

        if st.button("Run Labeled Retrieval Benchmark") and benchmark_k_values:
            try:
                benchmark_records = json.loads(benchmark_query_json)
                dataset = RetrievalBenchmarkDataset.from_records(
                    benchmark_records,
                    name=benchmark_name,
                    metadata={"source": "dashboard"},
                )
            except Exception as exc:
                st.error(f"Invalid labeled query JSON: {exc}")
            else:
                if not QDRANT_AVAILABLE:
                    st.error("Labeled retrieval benchmarking requires qdrant-client in this environment.")
                else:
                    exp = RetrievalExperiment(collection_name=benchmark_collection)
                    with st.spinner("Running retrieval benchmark..."):
                        results = exp.run_labeled_retrieval_benchmark(
                            dataset=dataset,
                            k_values=benchmark_k_values,
                        )

                    st.subheader("Per-Query Results")
                    st.dataframe(results)

                    if not results.empty:
                        st.subheader("Metric Summary by Top-K")
                        summary = (
                            results.groupby("top_k")[
                                [
                                    "hit_rate",
                                    "retrieval_accuracy",
                                    "recall_at_k",
                                    "precision_at_k",
                                    "mrr",
                                    "retrieval_latency",
                                    "retrieval_duration",
                                ]
                            ]
                            .mean()
                            .reset_index()
                        )
                        st.dataframe(summary)
                        st.line_chart(summary.set_index("top_k")[["recall_at_k", "hit_rate", "retrieval_latency", "retrieval_duration"]])

                        artifact_path = _save_dashboard_artifact(
                            artifact_type="labeled_retrieval_benchmark",
                            experiment_name=f"retrieval_{benchmark_name}",
                            results=results,
                            summary={
                                "num_queries": len(dataset.queries),
                                "num_k_values": len(benchmark_k_values),
                                "mean_hit_rate": float(summary["hit_rate"].mean()) if not summary.empty else 0.0,
                                "mean_recall_at_k": float(summary["recall_at_k"].mean()) if not summary.empty else 0.0,
                            },
                            config_payload={
                                "collection_name": benchmark_collection,
                                "dataset_name": benchmark_name,
                                "k_values": benchmark_k_values,
                            },
                            metadata={"dataset": dataset.to_dict() if hasattr(dataset, "to_dict") else None},
                        )
                        st.caption(f"Saved benchmark artifact: {artifact_path}")

with tab5:
    st.header("Corpus Ingestion")

    st.info("Index a local corpus into Qdrant with a reproducible manifest.")

    ingest_mode = st.radio("Source type", ["Local directory", "Uploaded files"], horizontal=True)
    corpus_preset = st.selectbox(
        "Corpus preset",
        ["Sample docs", "Custom path"],
        index=0,
    )
    collection_name = st.text_input(
        "Collection name",
        value=st.session_state.get("last_collection_name", "demo"),
        key="ingestion_collection_name",
    )
    embed_model = st.text_input("Embedding model", value=config.embed_model)
    chunk_size = st.number_input("Chunk size", min_value=64, max_value=4096, value=config.chunk.chunk_size, step=32)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1024, value=config.chunk.chunk_overlap, step=16)
    recreate_collection = st.checkbox("Recreate collection before ingest", value=True)

    source_input = None
    uploaded_files = []
    if ingest_mode == "Local directory":
        default_corpus_path = Path.cwd() / "data" / "docs"
        if corpus_preset == "Sample docs":
            source_input = str(default_corpus_path)
            st.caption(f"Using bundled sample corpus: {source_input}")
        else:
            source_input = st.text_input(
                "Corpus path",
                value=str(default_corpus_path if default_corpus_path.exists() else Path.cwd()),
                help="Path to a local directory or file on the machine running Streamlit.",
                key="custom_corpus_path",
            )
    else:
        uploaded_files = st.file_uploader(
            "Upload corpus files",
            accept_multiple_files=True,
            type=["txt", "md", "pdf", "docx"],
        )

    if ingest_mode == "Local directory" and source_input:
        preview_path = Path(source_input).expanduser()
        if preview_path.exists():
            try:
                supported_sources = preview_supported_sources(preview_path)
                st.caption(
                    f"Supported files found: {len(supported_sources)} "
                    f"({', '.join(path.name for path in supported_sources[:5]) or 'none'})"
                )
            except Exception as exc:
                st.caption(f"Corpus preview unavailable: {exc}")

    ingest_col, reset_col = st.columns(2)

    with ingest_col:
        build_clicked = st.button("Build Qdrant Index", type="primary")

    with reset_col:
        reset_clicked = st.button("Reset Collection")

    confirm_delete = False
    if reset_clicked:
        confirm_delete = st.checkbox(
            f"Confirm deletion of collection '{collection_name}'",
            value=False,
            key="confirm_delete_collection",
        )

    if build_clicked:
        try:
            if not QDRANT_AVAILABLE:
                st.error("Corpus ingestion requires qdrant-client in this environment.")
                ingest_source = None
            else:
                ingest_source = None
                if ingest_mode == "Uploaded files":
                    if not uploaded_files:
                        st.error("Upload at least one file before ingesting.")
                    else:
                        run_id = uuid4().hex
                        upload_dir = RESULTS_DIR / "uploads" / run_id
                        saved_files = _save_uploaded_files(uploaded_files, upload_dir)
                        ingest_source = upload_dir
                        st.write(f"Saved {len(saved_files)} uploaded file(s) to {upload_dir}")
                else:
                    ingest_source = Path(source_input).expanduser()

                if ingest_source is None:
                    pass
                elif ingest_mode == "Local directory" and not ingest_source.exists():
                    st.error(f"Path does not exist: {ingest_source}")
                else:
                    with st.spinner("Ingesting corpus into Qdrant..."):
                        result = ingest_corpus(
                            ingest_source,
                            collection_name=collection_name,
                            chunk_size=int(chunk_size),
                            chunk_overlap=int(chunk_overlap),
                            embed_model=embed_model,
                            recreate_collection=recreate_collection,
                        )

                    st.session_state["last_collection_name"] = collection_name
                    st.session_state["last_manifest_path"] = str(result.manifest_path)

                    st.success("Ingestion complete.")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Documents", result.manifest.total_documents)
                    col2.metric("Chunks", result.manifest.total_chunks)
                    col3.metric("Indexed Chunks", result.indexed_chunks)
                    col4.metric("Vector Size", result.manifest.vector_size or 0)

                    timing_cols = st.columns(4)
                    timing_cols[0].metric("Chunking Duration", f"{result.timings.get('chunking_duration', 0.0):.3f}s")
                    timing_cols[1].metric("Embedding Duration", f"{result.timings.get('embedding_duration', 0.0):.3f}s")
                    timing_cols[2].metric("Indexing Duration", f"{result.timings.get('indexing_duration', 0.0):.3f}s")
                    timing_cols[3].metric("Total Duration", f"{result.timings.get('total_duration', 0.0):.3f}s")

                    st.write("Manifest path:", result.manifest_path)
                    st.write("Collection info:", result.collection_info)
                    st.subheader("Manifest")
                    st.json(result.manifest.to_dict())
                    st.subheader("Timing Breakdown")
                    st.json(result.timings)

                    if result.chunks:
                        st.subheader("Sample Indexed Chunks")
                        preview_rows = []
                        for chunk in result.chunks[:10]:
                            preview_rows.append(
                                {
                                    "chunk_index": chunk.index,
                                    "start_char": chunk.start_char,
                                    "end_char": chunk.end_char,
                                    "text": chunk.text[:240],
                                    "source": chunk.metadata.get("source", ""),
                                    "file_hash": chunk.metadata.get("file_hash", ""),
                                }
                        )
                        st.dataframe(pd.DataFrame(preview_rows))
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")

    if reset_clicked and confirm_delete:
        if not QDRANT_AVAILABLE:
            st.error("Collection reset requires qdrant-client in this environment.")
        else:
            try:
                vector_store = VectorStore(collection_name=collection_name, embed_model=embed_model)
                vector_store.delete_collection()
                st.success(f"Collection '{collection_name}' was deleted.")
            except Exception as exc:
                st.error(f"Failed to delete collection: {exc}")
    elif reset_clicked and not confirm_delete:
        st.warning("Check the confirmation box to delete the collection.")

with tab6:
    st.header("Saved Benchmark Analytics")
    st.info("Load saved benchmark artifacts from `results/` and inspect them with the same charts used after each run.")

    artifact_paths = list_benchmark_artifacts(RESULTS_DIR)
    if not artifact_paths:
        st.warning("No saved benchmark artifacts found yet. Run a benchmark above to populate this view.")
    else:
        labels = [_artifact_label(path) for path in artifact_paths]
        default_index = 0
        last_artifact = st.session_state.get("last_benchmark_artifact_path")
        if last_artifact:
            for index, path in enumerate(artifact_paths):
                if str(path) == last_artifact:
                    default_index = index
                    break

        selected_label = st.selectbox("Artifact", labels, index=default_index)
        selected_path = artifact_paths[labels.index(selected_label)]
        artifact = load_benchmark_artifact(selected_path)
        st.session_state["last_benchmark_artifact_path"] = str(selected_path)

        st.caption(f"Artifact path: {selected_path}")

        metric_values = {
            key: value for key, value in artifact.summary.items() if isinstance(value, (int, float))
        }
        if metric_values:
            metric_columns = st.columns(min(4, len(metric_values)))
            for column, (key, value) in zip(metric_columns, metric_values.items()):
                column.metric(key.replace("_", " ").title(), f"{value:.3f}" if isinstance(value, float) else value)

        if artifact.config:
            with st.expander("Artifact Config", expanded=False):
                st.json(artifact.config)

        if artifact.metadata:
            with st.expander("Artifact Metadata", expanded=False):
                st.json(artifact.metadata)

        charts = create_benchmark_analytics_charts(artifact.results)
        if charts:
            st.subheader("Visual Analytics")
            for chart_name, chart in charts.items():
                if chart_name == "top_k_metrics":
                    st.line_chart(chart.set_index("top_k"))
                elif chart_name == "chunk_overlap_latency":
                    st.line_chart(chart.set_index("chunk_overlap"))
                elif chart_name == "chunking_strategy_metrics":
                    metric_cols = [col for col in chart.columns if col != "chunking_strategy"]
                    st.bar_chart(chart.set_index("chunking_strategy")[metric_cols])
                elif chart_name == "chunking_strategy_structure":
                    structure_cols = [col for col in chart.columns if col != "chunking_strategy"]
                    st.bar_chart(chart.set_index("chunking_strategy")[structure_cols])
                elif chart_name == "model_latency":
                    st.bar_chart(chart.set_index("model"))
                elif chart_name == "model_quality":
                    metric_cols = [col for col in chart.columns if col != "model"]
                    st.bar_chart(chart.set_index("model")[metric_cols])
                elif chart_name == "timing_columns":
                    x_col = next((col for col in ["model", "top_k", "chunking_strategy"] if col in chart.columns), None)
                    if x_col:
                        metric_cols = [col for col in chart.columns if col != x_col]
                        st.bar_chart(chart.set_index(x_col)[metric_cols])
                    else:
                        st.dataframe(chart)
                elif chart_name == "timing_breakdown":
                    x_col = next((col for col in ["model", "top_k", "chunking_strategy"] if col in chart.columns), None)
                    if x_col:
                        metric_cols = [col for col in chart.columns if col != x_col]
                        st.bar_chart(chart.set_index(x_col)[metric_cols])
                    else:
                        st.dataframe(chart)
                elif chart_name == "faithfulness":
                    index_col = "user_input" if "user_input" in chart.columns else "sample"
                    st.bar_chart(chart.set_index(index_col))
                elif chart_name == "score_distribution":
                    st.bar_chart(chart)
                else:
                    st.dataframe(chart)
        else:
            st.info("No chartable columns were found in this artifact.")

        if not artifact.results.empty:
            st.subheader("Raw Results")
            st.dataframe(artifact.results)

st.divider()
st.caption(f"RAG Benchmark Lab | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
