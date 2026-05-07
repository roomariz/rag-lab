from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ..config import config
from ..ingestion.chunker import Chunk, DocumentChunker
from ..retrieval import RetrievalPipeline
from ..benchmarks.timing import TimingBreakdown
from .artifacts import BenchmarkArtifact, save_benchmark_artifact
from .datasets import RetrievalBenchmarkDataset
from .retrieval_metrics import evaluate_retrieval_benchmark


@dataclass
class ChunkingStrategySpec:
    name: str
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkingStrategySpec":
        return cls(
            name=data["name"],
            chunk_size=int(data.get("chunk_size", 512)),
            chunk_overlap=int(data.get("chunk_overlap", 50)),
            separators=list(data["separators"]) if data.get("separators") is not None else None,
        )


@dataclass
class ChunkingBenchmarkResult:
    experiment_name: str
    dataset_name: str
    top_k: int
    per_query_results: pd.DataFrame
    summary: Dict[str, Any]
    strategy_summary: pd.DataFrame
    raw_strategy_results: pd.DataFrame = field(default_factory=pd.DataFrame)


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _parse_documents(documents: Sequence[str | Dict[str, Any]] | str) -> List[Dict[str, str]]:
    if isinstance(documents, str):
        blocks = [block.strip() for block in re.split(r"\n\s*\n", documents) if block.strip()]
        return [
            {
                "name": f"document_{index + 1}.txt",
                "text": block,
            }
            for index, block in enumerate(blocks or ([documents.strip()] if documents.strip() else []))
        ]

    parsed: List[Dict[str, str]] = []
    for index, document in enumerate(documents):
        if isinstance(document, dict):
            text = str(document.get("text", "")).strip()
            name = str(document.get("name", f"document_{index + 1}.txt")).strip() or f"document_{index + 1}.txt"
        else:
            text = str(document).strip()
            name = f"document_{index + 1}.txt"

        if text:
            parsed.append({"name": name, "text": text})

    return parsed


def _strategy_metadata(strategy: ChunkingStrategySpec) -> Dict[str, Any]:
    return {
        "chunking_strategy": strategy.name,
        "chunk_size": strategy.chunk_size,
        "chunk_overlap": strategy.chunk_overlap,
        "separators": strategy.separators or [],
    }


def _chunk_summary(strategy: ChunkingStrategySpec, chunks: List[Chunk]) -> Dict[str, Any]:
    chunk_lengths = [len(chunk.text) for chunk in chunks]
    separator_count = len(strategy.separators or [])

    return {
        "chunking_strategy": strategy.name,
        "chunk_size": strategy.chunk_size,
        "chunk_overlap": strategy.chunk_overlap,
        "num_separators": separator_count,
        "num_chunks": len(chunks),
        "total_chars": int(sum(chunk_lengths)),
        "avg_chunk_size": float(sum(chunk_lengths) / len(chunk_lengths)) if chunk_lengths else 0.0,
        "min_chunk_size": int(min(chunk_lengths)) if chunk_lengths else 0,
        "max_chunk_size": int(max(chunk_lengths)) if chunk_lengths else 0,
    }


def _make_collection_name(prefix: str, strategy_name: str, run_id: str) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", strategy_name).strip("_").lower()
    return f"{prefix}_{safe_name}_{run_id.lower()}"


def run_chunking_quality_benchmark(
    documents: Sequence[str | Dict[str, Any]] | str,
    dataset: RetrievalBenchmarkDataset,
    strategies: Sequence[ChunkingStrategySpec | Dict[str, Any]],
    *,
    top_k: int = 5,
    embed_model: Optional[str] = None,
    collection_prefix: str = "chunking_benchmark",
    cleanup_collections: bool = True,
) -> ChunkingBenchmarkResult:
    docs = _parse_documents(documents)
    if not docs:
        raise ValueError("No documents were provided for chunking evaluation.")
    if not dataset.queries:
        raise ValueError("Chunking benchmark requires a labeled query dataset.")
    if not strategies:
        raise ValueError("At least one chunking strategy is required.")

    run_id = _utc_run_id()
    embed_model = embed_model or config.embed_model

    strategy_results: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    for raw_strategy in strategies:
        strategy = raw_strategy if isinstance(raw_strategy, ChunkingStrategySpec) else ChunkingStrategySpec.from_dict(raw_strategy)
        chunker = DocumentChunker(
            chunk_size=strategy.chunk_size,
            chunk_overlap=strategy.chunk_overlap,
            separators=strategy.separators,
        )

        all_chunks: List[Chunk] = []
        chunking_duration = 0.0
        for index, document in enumerate(docs):
            chunk_start = time.perf_counter()
            chunks = chunker.chunk_text(
                document["text"],
                metadata={
                    "document_index": index,
                    "document_name": document["name"],
                    "source": document["name"],
                    "filename": Path(document["name"]).name,
                    "document_hash": hashlib.sha256(document["text"].encode("utf-8")).hexdigest(),
                    **_strategy_metadata(strategy),
                },
            )
            chunking_duration += time.perf_counter() - chunk_start
            all_chunks.extend(chunks)

        collection_name = _make_collection_name(collection_prefix, strategy.name, run_id)
        pipeline = RetrievalPipeline(
            collection_name=collection_name,
            top_k=top_k,
            embed_model=embed_model,
        )

        pipeline.vector_store.delete_collection()
        indexing_result = pipeline.vector_store.add_chunks(
            all_chunks,
            run_id=run_id,
            extra_payload={
                "collection_name": collection_name,
                "embed_model": embed_model,
                **_strategy_metadata(strategy),
            },
        )
        indexing_timings = TimingBreakdown.from_mapping(indexing_result.get("timings", {}))

        benchmark = evaluate_retrieval_benchmark(dataset, pipeline)
        frame = benchmark.per_query_results.copy()
        frame["experiment_name"] = benchmark.experiment_name
        frame["dataset_name"] = benchmark.dataset_name
        frame["chunking_strategy"] = strategy.name
        frame["chunk_size"] = strategy.chunk_size
        frame["chunk_overlap"] = strategy.chunk_overlap
        frame["num_documents"] = len(docs)
        frame["num_chunks"] = len(all_chunks)
        frame["avg_chunk_size"] = float(sum(len(chunk.text) for chunk in all_chunks) / len(all_chunks)) if all_chunks else 0.0
        frame["chunking_duration"] = chunking_duration
        frame["embedding_duration"] = indexing_timings.embedding_duration
        frame["indexing_duration"] = indexing_timings.indexing_duration
        frame["strategy_recall_at_k"] = benchmark.summary.get("mean_recall_at_k", 0.0)
        frame["strategy_precision_at_k"] = benchmark.summary.get("mean_precision_at_k", 0.0)
        frame["strategy_hit_rate"] = benchmark.summary.get("mean_hit_rate", 0.0)
        frame["strategy_mrr"] = benchmark.summary.get("mean_mrr", 0.0)
        frame["strategy_retrieval_latency"] = benchmark.summary.get("mean_retrieval_latency", 0.0)
        frame["strategy_retrieval_duration"] = benchmark.summary.get("mean_retrieval_duration", 0.0)
        frame["strategy_total_duration"] = chunking_duration + indexing_timings.total_duration
        strategy_results.append(frame)

        summary_rows.append({
            **_chunk_summary(strategy, all_chunks),
            "dataset_name": dataset.name,
            "top_k": top_k,
            "mean_hit_rate": benchmark.summary.get("mean_hit_rate", 0.0),
            "mean_recall_at_k": benchmark.summary.get("mean_recall_at_k", 0.0),
            "mean_precision_at_k": benchmark.summary.get("mean_precision_at_k", 0.0),
            "mean_mrr": benchmark.summary.get("mean_mrr", 0.0),
            "mean_retrieval_accuracy": benchmark.summary.get("mean_retrieval_accuracy", 0.0),
            "mean_retrieval_latency": benchmark.summary.get("mean_retrieval_latency", 0.0),
            "mean_retrieval_duration": benchmark.summary.get("mean_retrieval_duration", 0.0),
            "chunking_duration": chunking_duration,
            "embedding_duration": indexing_timings.embedding_duration,
            "indexing_duration": indexing_timings.indexing_duration,
            "total_duration": chunking_duration + indexing_timings.total_duration,
            "run_id": run_id,
            "collection_name": collection_name,
            "embed_model": embed_model,
        })

        if cleanup_collections:
            pipeline.vector_store.delete_collection()

    per_query_results = pd.concat(strategy_results, ignore_index=True) if strategy_results else pd.DataFrame()
    strategy_summary = pd.DataFrame(summary_rows)

    summary: Dict[str, Any] = {
        "dataset_name": dataset.name,
        "top_k": top_k,
        "num_strategies": float(len(summary_rows)),
        "mean_chunking_duration": float(strategy_summary["chunking_duration"].mean()) if not strategy_summary.empty else 0.0,
        "mean_embedding_duration": float(strategy_summary["embedding_duration"].mean()) if not strategy_summary.empty else 0.0,
        "mean_indexing_duration": float(strategy_summary["indexing_duration"].mean()) if not strategy_summary.empty else 0.0,
        "mean_hit_rate": float(strategy_summary["mean_hit_rate"].mean()) if not strategy_summary.empty else 0.0,
        "mean_recall_at_k": float(strategy_summary["mean_recall_at_k"].mean()) if not strategy_summary.empty else 0.0,
        "mean_precision_at_k": float(strategy_summary["mean_precision_at_k"].mean()) if not strategy_summary.empty else 0.0,
        "mean_mrr": float(strategy_summary["mean_mrr"].mean()) if not strategy_summary.empty else 0.0,
        "mean_retrieval_latency": float(strategy_summary["mean_retrieval_latency"].mean()) if not strategy_summary.empty else 0.0,
        "mean_retrieval_duration": float(strategy_summary["mean_retrieval_duration"].mean()) if not strategy_summary.empty else 0.0,
        "mean_total_duration": float(strategy_summary["total_duration"].mean()) if not strategy_summary.empty else 0.0,
    }

    return ChunkingBenchmarkResult(
        experiment_name=f"chunking_{dataset.name}",
        dataset_name=dataset.name,
        top_k=top_k,
        per_query_results=per_query_results,
        summary=summary,
        strategy_summary=strategy_summary,
        raw_strategy_results=per_query_results,
    )


def save_chunking_benchmark(
    result: ChunkingBenchmarkResult,
    *,
    strategies: Sequence[ChunkingStrategySpec | Dict[str, Any]],
    documents: Sequence[str | Dict[str, Any]] | str,
    filename: Optional[str] = None,
) -> Path:
    artifact = BenchmarkArtifact(
        artifact_type="chunking_quality_benchmark",
        experiment_name=result.experiment_name,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config={
            "top_k": result.top_k,
            "dataset_name": result.dataset_name,
            "strategies": [
                strategy.__dict__ if isinstance(strategy, ChunkingStrategySpec) else dict(strategy)
                for strategy in strategies
            ],
        },
        summary=result.summary,
        results=result.per_query_results,
        metadata={
            "documents": _parse_documents(documents),
            "strategy_summary_csv": result.strategy_summary.to_csv(index=False),
        },
    )
    return save_benchmark_artifact(artifact, filename=filename)
