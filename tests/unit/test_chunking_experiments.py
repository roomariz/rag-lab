import pandas as pd

from src.benchmarks.chunking_experiments import (
    ChunkingBenchmarkResult,
    ChunkingStrategySpec,
    _parse_documents,
    save_chunking_benchmark,
)
from src.benchmarks.artifacts import load_benchmark_artifact
from src.visualization.benchmark_charts import create_benchmark_analytics_charts


def test_parse_documents_keeps_names_for_named_records():
    docs = _parse_documents(
        [
            {"name": "docker.md", "text": "Docker containers."},
            {"name": "kubernetes.md", "text": "Kubernetes clusters."},
        ]
    )

    assert docs[0]["name"] == "docker.md"
    assert docs[1]["text"] == "Kubernetes clusters."


def test_save_chunking_benchmark_round_trip(tmp_path):
    result = ChunkingBenchmarkResult(
        experiment_name="chunking_demo",
        dataset_name="demo_dataset",
        top_k=5,
        per_query_results=pd.DataFrame(
            [
                {
                    "query": "What is Docker?",
                    "chunking_strategy": "balanced",
                    "chunk_size": 512,
                    "chunk_overlap": 64,
                    "mean_recall_at_k": 1.0,
                }
            ]
        ),
        summary={"mean_recall_at_k": 1.0, "num_strategies": 1.0},
        strategy_summary=pd.DataFrame(
            [
                {
                    "chunking_strategy": "balanced",
                    "chunk_size": 512,
                    "chunk_overlap": 64,
                    "num_chunks": 4,
                    "avg_chunk_size": 128.0,
                    "mean_recall_at_k": 1.0,
                }
            ]
        ),
    )

    path = save_chunking_benchmark(
        result,
        strategies=[ChunkingStrategySpec(name="balanced", chunk_size=512, chunk_overlap=64)],
        documents=[{"name": "docker.md", "text": "Docker containers."}],
        filename="chunking_demo.json",
    )

    artifact = load_benchmark_artifact(path)

    assert artifact.artifact_type == "chunking_quality_benchmark"
    assert artifact.summary["mean_recall_at_k"] == 1.0
    assert artifact.metadata["documents"][0]["name"] == "docker.md"
    assert "chunking_strategy" in artifact.results.columns


def test_create_benchmark_analytics_charts_handles_chunking_strategy():
    frame = pd.DataFrame(
        [
            {
                "chunking_strategy": "small",
                "mean_recall_at_k": 0.5,
                "mean_precision_at_k": 0.25,
                "mean_hit_rate": 1.0,
                "mean_mrr": 0.5,
                "num_chunks": 8,
                "avg_chunk_size": 120.0,
                "chunk_size": 256,
                "chunk_overlap": 32,
            },
            {
                "chunking_strategy": "large",
                "mean_recall_at_k": 1.0,
                "mean_precision_at_k": 0.5,
                "mean_hit_rate": 1.0,
                "mean_mrr": 1.0,
                "num_chunks": 4,
                "avg_chunk_size": 240.0,
                "chunk_size": 512,
                "chunk_overlap": 64,
            },
        ]
    )

    charts = create_benchmark_analytics_charts(frame)

    assert "chunking_strategy_metrics" in charts
    assert "chunking_strategy_structure" in charts
    assert list(charts["chunking_strategy_metrics"]["chunking_strategy"]) == ["large", "small"]
