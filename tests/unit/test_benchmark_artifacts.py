import pandas as pd

from src.benchmarks.artifacts import (
    BenchmarkArtifact,
    list_benchmark_artifacts,
    load_benchmark_artifact,
    save_benchmark_artifact,
)
from src.visualization.benchmark_charts import create_benchmark_analytics_charts


def test_save_and_load_benchmark_artifact_round_trip(tmp_path):
    artifact = BenchmarkArtifact(
        artifact_type="labeled_retrieval_benchmark",
        experiment_name="retrieval_demo",
        timestamp="2026-05-06T12:30:00",
        config={"collection_name": "demo"},
        summary={"mean_hit_rate": 1.0},
        results=pd.DataFrame(
            [
                {"top_k": 1, "hit_rate": 1.0, "retrieval_latency": 0.2},
                {"top_k": 3, "hit_rate": 0.0, "retrieval_latency": 0.4},
            ]
        ),
        metadata={"source": "test"},
    )

    path = save_benchmark_artifact(artifact, results_dir=tmp_path)
    loaded = load_benchmark_artifact(path)

    assert loaded.artifact_type == artifact.artifact_type
    assert loaded.experiment_name == artifact.experiment_name
    assert loaded.summary == artifact.summary
    assert loaded.config == artifact.config
    assert loaded.metadata == artifact.metadata
    assert loaded.results.equals(artifact.results)


def test_list_benchmark_artifacts_includes_json_and_csv(tmp_path):
    (tmp_path / "run_a.json").write_text("{}")
    (tmp_path / "run_b.csv").write_text("x\n1\n")

    paths = list_benchmark_artifacts(tmp_path)

    assert [path.name for path in paths] == ["run_b.csv", "run_a.json"]


def test_create_benchmark_analytics_charts_groups_top_k_metrics():
    frame = pd.DataFrame(
        [
            {"top_k": 1, "hit_rate": 1.0, "retrieval_latency": 0.2},
            {"top_k": 1, "hit_rate": 0.0, "retrieval_latency": 0.3},
            {"top_k": 3, "hit_rate": 1.0, "retrieval_latency": 0.4},
        ]
    )

    charts = create_benchmark_analytics_charts(frame)

    assert "top_k_metrics" in charts
    assert list(charts["top_k_metrics"]["top_k"]) == [1, 3]
