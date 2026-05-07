import pandas as pd

from src.benchmarks.artifacts import load_benchmark_artifact
from src.benchmarks.embedding_benchmarks import (
    EmbeddingComparisonResult,
    compare_embedding_models,
    run_embedding_comparison,
    save_embedding_comparison,
)
from src.retrieval.vector_store import RetrievedChunk


def test_compare_embedding_models_preserves_latency_columns(monkeypatch):
    import src.benchmarks.embedding_benchmarks as embedding_benchmarks

    def fake_latency(self, texts, num_runs=3):
        base = 1.0 if self.embed_model == "model-a" else 2.0
        return {
            "mean_latency": base,
            "min_latency": base,
            "max_latency": base,
            "num_texts": len(texts),
        }

    monkeypatch.setattr(embedding_benchmarks.EmbeddingBenchmark, "benchmark_latency", fake_latency)

    frame = compare_embedding_models(
        texts=["alpha", "beta"],
        models=["model-a", "model-b"],
        num_runs=2,
    )

    assert list(frame["model"]) == ["model-a", "model-b"]
    assert list(frame["mean_latency"]) == [1.0, 2.0]
    assert "document_embedding_latency" in frame.columns


def test_run_embedding_comparison_aggregates_quality_metrics(monkeypatch):
    import src.benchmarks.embedding_benchmarks as embedding_benchmarks

    class FakeVectorStore:
        def __init__(self, collection_name, embed_model=None):
            self.collection_name = collection_name
            self.embed_model = embed_model

        def add_chunks(self, chunks, **kwargs):
            return {"indexed": len(chunks)}

        def delete_collection(self):
            return None

    class FakePipeline:
        def __init__(self, collection_name="demo", top_k=5, embed_model=None):
            self.top_k = top_k
            self.embed_model = embed_model
            self.vector_store = type(
                "Store",
                (),
                {"collection_name": collection_name, "embed_model": embed_model},
            )()
            self._client = object()

        def run(self, query, include_generation=True):
            return type(
                "Result",
                (),
                {
                    "retrieved_chunks": [
                        RetrievedChunk(
                            text="Docker containers.",
                            score=0.9,
                            metadata={"filename": "docker.md", "source": "docker.md"},
                            index=0,
                            point_id="docker-1",
                        )
                    ],
                    "generated_response": "Docker containers run applications.",
                    "retrieval_latency": 0.1,
                    "generation_latency": 0.2 if include_generation else 0.0,
                    "total_latency": 0.3 if include_generation else 0.1,
                },
            )()

    class FakeRAGEvaluator:
        def __init__(self, llm_model=None, embed_model=None):
            self.llm_model = llm_model
            self.embed_model = embed_model

        def evaluate(self, queries, retrieved_contexts, generated_responses, references):
            return type(
                "RagasResult",
                (),
                {
                    "metrics": {
                        "faithfulness": 0.75,
                        "answer_relevancy": 0.8,
                        "context_precision": 0.9,
                        "context_recall": 0.85,
                    },
                    "per_sample_results": pd.DataFrame(
                        [
                            {
                                "faithfulness": 0.75,
                                "answer_relevancy": 0.8,
                                "context_precision": 0.9,
                                "context_recall": 0.85,
                            }
                        ]
                    ),
                },
            )()

    def fake_latency(self, texts, num_runs=3):
        return {
            "mean_latency": 0.4,
            "min_latency": 0.2,
            "max_latency": 0.6,
            "num_texts": len(texts),
        }

    monkeypatch.setattr(embedding_benchmarks, "QDRANT_AVAILABLE", True)
    monkeypatch.setattr(embedding_benchmarks, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(embedding_benchmarks, "RetrievalPipeline", FakePipeline)
    monkeypatch.setattr(embedding_benchmarks, "RAGEvaluator", FakeRAGEvaluator)
    monkeypatch.setattr(embedding_benchmarks.EmbeddingBenchmark, "benchmark_latency", fake_latency)

    result = run_embedding_comparison(
        texts=[{"name": "docker.md", "text": "Docker containers."}],
        models=["model-a"],
        num_runs=2,
        query_records=[
            {
                "query": "What is Docker?",
                "relevant_sources": ["docker.md"],
                "reference": "Docker containers run applications.",
            }
        ],
        top_k=5,
    )

    assert isinstance(result, EmbeddingComparisonResult)
    assert list(result.per_model_results["model"]) == ["model-a"]
    row = result.per_model_results.iloc[0]
    assert row["mean_recall_at_k"] == 1.0
    assert row["mean_faithfulness"] == 0.75
    assert bool(row["ragas_enabled"]) is True
    assert not result.per_query_results.empty
    assert not result.per_sample_results.empty


def test_save_embedding_comparison_round_trip(tmp_path, monkeypatch):
    import src.benchmarks.embedding_benchmarks as embedding_benchmarks
    from src.benchmarks.artifacts import save_benchmark_artifact as real_save_benchmark_artifact

    result = EmbeddingComparisonResult(
        experiment_name="embedding_comparison",
        timestamp="2026-05-07T10:00:00",
        summary={"num_models": 1.0, "mean_latency": 0.5},
        per_model_results=pd.DataFrame(
            [
                {
                    "model": "model-a",
                    "mean_latency": 0.5,
                    "mean_recall_at_k": 1.0,
                }
            ]
        ),
        per_query_results=pd.DataFrame([{"model": "model-a", "query": "What is Docker?"}]),
        per_sample_results=pd.DataFrame([{"model": "model-a", "faithfulness": 0.75}]),
    )

    def fake_save(artifact, filename=None):
        return real_save_benchmark_artifact(artifact, filename=filename, results_dir=tmp_path)

    monkeypatch.setattr(embedding_benchmarks, "save_benchmark_artifact", fake_save)

    path = save_embedding_comparison(result, filename="embedding_comparison.json")
    artifact = load_benchmark_artifact(path)

    assert artifact.artifact_type == "embedding_comparison"
    assert artifact.summary["num_models"] == 1.0
    assert artifact.metadata["per_query_results"][0]["model"] == "model-a"
    assert "mean_recall_at_k" in artifact.results.columns
