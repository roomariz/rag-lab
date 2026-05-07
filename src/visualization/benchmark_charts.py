from __future__ import annotations

from typing import Dict

import pandas as pd

from ..benchmarks.timing import TIMING_FIELDS


def create_benchmark_analytics_charts(results: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    charts: Dict[str, pd.DataFrame] = {}

    if results.empty:
        return charts

    metric_aliases = [
        "hit_rate",
        "retrieval_accuracy",
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "retrieval_latency",
        "mean_hit_rate",
        "mean_retrieval_accuracy",
        "mean_recall_at_k",
        "mean_precision_at_k",
        "mean_mrr",
        "mean_retrieval_latency",
    ]
    metric_columns = [column for column in metric_aliases if column in results.columns]
    if "top_k" in results.columns and metric_columns:
        charts["top_k_metrics"] = results.groupby("top_k")[metric_columns].mean().reset_index()

    if "chunk_overlap" in results.columns and "retrieval_latency" in results.columns:
        charts["chunk_overlap_latency"] = (
            results.groupby("chunk_overlap")["retrieval_latency"].mean().reset_index()
        )

    if "chunking_strategy" in results.columns and metric_columns:
        charts["chunking_strategy_metrics"] = (
            results.groupby("chunking_strategy")[metric_columns].mean().reset_index()
        )

    if "chunking_strategy" in results.columns and "num_chunks" in results.columns:
        charts["chunking_strategy_structure"] = (
            results.groupby("chunking_strategy")[["num_chunks", "avg_chunk_size", "chunk_size", "chunk_overlap"]]
            .mean()
            .reset_index()
        )

    if "model" in results.columns and "mean_latency" in results.columns:
        charts["model_latency"] = results[["model", "mean_latency"]].copy()

    model_metric_columns = [
        column
        for column in [
            "document_embedding_latency",
            "query_embedding_latency",
            "mean_latency",
            "mean_recall_at_k",
            "mean_faithfulness",
            "mean_retrieval_latency",
            "mean_hit_rate",
            "mean_precision_at_k",
            "mean_mrr",
        ]
        if column in results.columns
    ]
    if "model" in results.columns and len(model_metric_columns) > 1:
        charts["model_quality"] = results[["model", *model_metric_columns]].copy()

    timing_columns = [column for column in TIMING_FIELDS if column in results.columns]
    if timing_columns:
        chart_columns = [column for column in ["model", "top_k", "chunking_strategy"] if column in results.columns]
        if chart_columns:
            charts["timing_columns"] = results[[*chart_columns[:1], *timing_columns]].copy()
        else:
            charts["timing_columns"] = results[timing_columns].copy()

    if "faithfulness" in results.columns:
        frame = results.copy()
        if "user_input" not in frame.columns:
            frame = frame.reset_index().rename(columns={"index": "sample"})
            charts["faithfulness"] = frame[["sample", "faithfulness"]]
        else:
            charts["faithfulness"] = frame[["user_input", "faithfulness"]]

    if "score" in results.columns:
        charts["score_distribution"] = results[["score"]].copy()

    return charts
