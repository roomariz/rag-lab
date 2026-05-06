import streamlit as st
import pandas as pd
import time
import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import tempfile

from src.config import config, RESULTS_DIR
from src.retrieval import RetrievalPipeline, create_test_queries
from src.ingestion import DocumentChunker, ingest_corpus
from src.benchmarks import (
    RAGEvaluator,
    compare_embedding_models,
    RetrievalExperiment,
    RetrievalBenchmarkDataset,
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

        top_k = st.slider("Top K", 1, 10, 5)
        score_threshold = st.slider("Score Threshold", 0.0, 1.0, 0.0, 0.1)

        run_btn = st.button("Run Retrieval", type="primary")

    with col2:
        if run_btn and query:
            pipeline = RetrievalPipeline(
                collection_name="demo",
                top_k=top_k,
                score_threshold=score_threshold,
            )

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
                metadata={"latencies": result.latencies, "timestamp": result.timestamp},
            )
            st.caption(f"Saved benchmark artifact: {artifact_path}")

with tab3:
    st.header("Embedding Model Comparison")

    if compare_embedding_models is None:
        st.warning("Embedding comparison dependencies are unavailable in this environment.")
    else:
        st.info("Compare embedding model latency and quality across different models.")

        test_texts = st.text_area(
            "Test texts (one per line):",
            value="Docker is a platform for containerization.\nKubernetes orchestrates containers.\nSQL injection is a vulnerability.",
        ).strip().split("\n")

        models = st.multiselect(
            "Models to compare:",
            ["nomic-embed-text:latest", "mxbai-embed-large:latest", "bge-m3:latest"],
            default=["nomic-embed-text:latest"],
        )

        num_runs = st.slider("Number of runs", 1, 10, 3)

        if st.button("Run Comparison", type="primary") and models and test_texts:
            with st.spinner("Running benchmarks..."):
                results = compare_embedding_models(
                    texts=test_texts,
                    models=models,
                    num_runs=num_runs,
                )

            st.subheader("Results")
            st.dataframe(results)

            st.subheader("Latency Comparison")
            st.line_chart(results.set_index("model")["mean_latency"])

            artifact_path = _save_dashboard_artifact(
                artifact_type="embedding_comparison",
                experiment_name="embedding_comparison",
                results=results,
                summary={
                    "num_models": len(models),
                    "num_texts": len(test_texts),
                    "mean_latency": float(results["mean_latency"].mean()) if not results.empty else 0.0,
                },
                config_payload={
                    "models": models,
                    "num_runs": num_runs,
                    "text_count": len(test_texts),
                },
            )
            st.caption(f"Saved benchmark artifact: {artifact_path}")

with tab4:
    st.header("Retrieval Experiments")

    st.info("Run experiments to tune retrieval parameters.")

    experiment_type = st.selectbox(
        "Experiment Type",
        ["Top-K Tuning", "Chunk Overlap Tuning", "Labeled Retrieval Benchmark", "Hybrid Retrieval"],
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

            exp = RetrievalExperiment(collection_name="topk_tune")
            query_data = [{"question": q} for q in queries]

            with st.spinner("Running experiment..."):
                results = exp.run_top_k_tuning(query_data, k_values)

            st.subheader("Results")
            st.dataframe(results)

            st.subheader("Latency by K")
            latency_by_k = results.groupby("top_k")["retrieval_latency"].mean()
            st.line_chart(latency_by_k)

            artifact_path = _save_dashboard_artifact(
                artifact_type="top_k_tuning",
                experiment_name="top_k_tuning",
                results=results,
                summary={
                    "num_queries": len(queries),
                    "num_k_values": len(k_values),
                    "mean_retrieval_latency": float(results["retrieval_latency"].mean()) if not results.empty else 0.0,
                },
                config_payload={
                    "collection_name": exp.collection_name,
                    "k_values": k_values,
                },
            )
            st.caption(f"Saved benchmark artifact: {artifact_path}")

    elif experiment_type == "Chunk Overlap Tuning":
        st.subheader("Chunk Overlap Tuning Experiment")

        st.info("This experiment requires a document collection to be pre-loaded.")

        documents = st.text_area(
            "Test documents:",
            value="Sample document text for testing chunking strategies.",
        ).strip()

        queries = st.text_area("Queries:").strip().split("\n")
        overlap_values = st.multiselect("Overlap values:", [0, 25, 50, 100, 200], default=[0, 50, 100])

        st.warning("Document ingestion required before running this experiment.")

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
                            ]
                        ]
                        .mean()
                        .reset_index()
                    )
                    st.dataframe(summary)
                    st.line_chart(summary.set_index("top_k")[["recall_at_k", "hit_rate", "retrieval_latency"]])

with tab5:
    st.header("Corpus Ingestion")

    st.info("Index a local corpus into Qdrant with a reproducible manifest.")

    ingest_mode = st.radio("Source type", ["Local directory", "Uploaded files"], horizontal=True)
    collection_name = st.text_input("Collection name", value=st.session_state.get("last_collection_name", "demo"))
    embed_model = st.text_input("Embedding model", value=config.embed_model)
    chunk_size = st.number_input("Chunk size", min_value=64, max_value=4096, value=config.chunk.chunk_size, step=32)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1024, value=config.chunk.chunk_overlap, step=16)
    recreate_collection = st.checkbox("Recreate collection before ingest", value=True)

    source_input = None
    uploaded_files = []
    if ingest_mode == "Local directory":
        source_input = st.text_input(
            "Corpus path",
            value=str(Path.cwd()),
            help="Path to a local directory or file on the machine running Streamlit.",
        )
    else:
        uploaded_files = st.file_uploader(
            "Upload corpus files",
            accept_multiple_files=True,
            type=["txt", "md", "pdf", "docx"],
        )

    if st.button("Build Qdrant Index", type="primary"):
        try:
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

                st.write("Manifest path:", result.manifest_path)
                st.write("Collection info:", result.collection_info)
                st.subheader("Manifest")
                st.json(result.manifest.to_dict())

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

st.divider()
st.caption(f"RAG Benchmark Lab | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
