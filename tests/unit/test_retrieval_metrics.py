import pandas as pd
import pytest

from src.benchmarks.datasets import RetrievalBenchmarkDataset, RetrievalQuery
from src.benchmarks.retrieval_metrics import (
    evaluate_query_retrieval,
    summarize_retrieval_metrics,
)
from src.retrieval.vector_store import RetrievedChunk


def test_evaluate_query_retrieval_scores_relevant_hits():
    query = RetrievalQuery(
        query="What is Docker?",
        relevant_sources=["docker.md"],
    )
    retrieved_chunks = [
        RetrievedChunk(text="intro", score=0.9, metadata={"filename": "docker.md"}, index=0, point_id="a"),
        RetrievedChunk(text="other", score=0.1, metadata={"filename": "misc.md"}, index=1, point_id="b"),
    ]

    row = evaluate_query_retrieval(query, retrieved_chunks, top_k=2)

    assert row["hit_rate"] == 1.0
    assert row["retrieval_accuracy"] == 1.0
    assert row["recall_at_k"] == 1.0
    assert row["precision_at_k"] == 0.5
    assert row["mrr"] == 1.0
    assert row["first_relevant_rank"] == 1


def test_summarize_retrieval_metrics_aggregates_means():
    df = pd.DataFrame(
        [
            {"hit_rate": 1.0, "retrieval_accuracy": 1.0, "recall_at_k": 0.5, "precision_at_k": 0.5, "mrr": 1.0, "retrieval_latency": 0.1},
            {"hit_rate": 0.0, "retrieval_accuracy": 0.0, "recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0, "retrieval_latency": 0.2},
        ]
    )

    summary = summarize_retrieval_metrics(df)

    assert summary["num_queries"] == 2.0
    assert summary["mean_hit_rate"] == 0.5
    assert summary["mean_recall_at_k"] == 0.25
    assert summary["mean_retrieval_latency"] == pytest.approx(0.15)
