from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from ..retrieval.vector_store import RetrievedChunk
from .datasets import RetrievalBenchmarkDataset, RetrievalQuery


@dataclass
class RetrievalBenchmarkResult:
    experiment_name: str
    dataset_name: str
    top_k: int
    per_query_results: pd.DataFrame
    summary: Dict[str, float]


def _normalize(values: Sequence[str]) -> set[str]:
    return {str(value) for value in values if value is not None and str(value)}


def _chunk_sources(chunk: RetrievedChunk) -> set[str]:
    metadata = chunk.metadata or {}
    sources = {
        str(metadata.get("source", "")),
        str(metadata.get("filename", "")),
        str(metadata.get("file_path", "")),
        str(metadata.get("path", "")),
    }
    return {value for value in sources if value}


def is_relevant(chunk: RetrievedChunk, query: RetrievalQuery) -> bool:
    relevant_ids = _normalize(query.relevant_ids)
    if chunk.point_id and chunk.point_id in relevant_ids:
        return True

    relevant_sources = _normalize(query.relevant_sources)
    if relevant_sources and _chunk_sources(chunk) & relevant_sources:
        return True

    return False


def evaluate_query_retrieval(
    query: RetrievalQuery,
    retrieved_chunks: Sequence[RetrievedChunk],
    top_k: int,
) -> Dict[str, Any]:
    relevance_flags = [is_relevant(chunk, query) for chunk in retrieved_chunks]
    relevant_hits = sum(relevance_flags)
    relevant_total = len(query.relevant_ids) or len(query.relevant_sources)
    first_rank = next((index + 1 for index, flag in enumerate(relevance_flags) if flag), None)

    recall = relevant_hits / relevant_total if relevant_total else 0.0
    precision = relevant_hits / max(top_k, 1)
    hit_rate = 1.0 if relevant_hits > 0 else 0.0
    retrieval_accuracy = hit_rate
    mrr = 1.0 / first_rank if first_rank else 0.0

    retrieved_ids = [chunk.point_id for chunk in retrieved_chunks]
    retrieved_sources = [
        sorted(_chunk_sources(chunk))[0] if _chunk_sources(chunk) else ""
        for chunk in retrieved_chunks
    ]

    return {
        "query": query.query,
        "top_k": top_k,
        "relevant_total": relevant_total,
        "relevant_hits": relevant_hits,
        "hit_rate": hit_rate,
        "retrieval_accuracy": retrieval_accuracy,
        "recall_at_k": recall,
        "precision_at_k": precision,
        "mrr": mrr,
        "first_relevant_rank": first_rank,
        "retrieved_ids": retrieved_ids,
        "retrieved_sources": retrieved_sources,
        "relevance_flags": relevance_flags,
        "retrieved_count": len(retrieved_chunks),
    }


def summarize_retrieval_metrics(per_query_results: pd.DataFrame) -> Dict[str, float]:
    if per_query_results.empty:
        return {
            "num_queries": 0.0,
            "mean_hit_rate": 0.0,
            "mean_retrieval_accuracy": 0.0,
            "mean_recall_at_k": 0.0,
            "mean_precision_at_k": 0.0,
            "mean_mrr": 0.0,
            "mean_retrieval_latency": 0.0,
            "mean_retrieval_duration": 0.0,
        }

    summary = {
        "num_queries": float(len(per_query_results)),
        "mean_hit_rate": float(per_query_results["hit_rate"].mean()),
        "mean_retrieval_accuracy": float(per_query_results["retrieval_accuracy"].mean()),
        "mean_recall_at_k": float(per_query_results["recall_at_k"].mean()),
        "mean_precision_at_k": float(per_query_results["precision_at_k"].mean()),
        "mean_mrr": float(per_query_results["mrr"].mean()),
    }

    if "retrieval_latency" in per_query_results.columns:
        summary["mean_retrieval_latency"] = float(per_query_results["retrieval_latency"].mean())
    else:
        summary["mean_retrieval_latency"] = 0.0

    if "retrieval_duration" in per_query_results.columns:
        summary["mean_retrieval_duration"] = float(per_query_results["retrieval_duration"].mean())
    else:
        summary["mean_retrieval_duration"] = summary["mean_retrieval_latency"]

    return summary


def evaluate_retrieval_benchmark(
    dataset: RetrievalBenchmarkDataset,
    pipeline,
) -> RetrievalBenchmarkResult:
    rows: List[Dict[str, Any]] = []

    for query in dataset.queries:
        result = pipeline.run(query.query, include_generation=False)
        row = evaluate_query_retrieval(query, result.retrieved_chunks, pipeline.top_k)
        row["retrieval_latency"] = result.retrieval_latency
        row["retrieval_duration"] = result.timings.get("retrieval_duration", result.retrieval_latency)
        row["collection_name"] = pipeline.vector_store.collection_name
        row["embed_model"] = pipeline.vector_store.embed_model
        rows.append(row)

    per_query_results = pd.DataFrame(rows)
    summary = summarize_retrieval_metrics(per_query_results)

    return RetrievalBenchmarkResult(
        experiment_name=f"retrieval_{dataset.name}",
        dataset_name=dataset.name,
        top_k=pipeline.top_k,
        per_query_results=per_query_results,
        summary=summary,
    )
