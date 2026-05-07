from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency guard
    OpenAI = None

from ..config import config
from ..ingestion.chunker import DocumentChunker
from ..retrieval.pipeline import RetrievalPipeline
from ..retrieval.vector_store import QDRANT_AVAILABLE, VectorStore
from .artifacts import BenchmarkArtifact, save_benchmark_artifact
from .datasets import RetrievalQuery
from .retrieval_metrics import evaluate_query_retrieval, summarize_retrieval_metrics


try:
    from .evaluator import RAGEvaluator
except ImportError:  # pragma: no cover - optional dependency guard
    RAGEvaluator = None


class OllamaEmbeddings:
    def __init__(self, embed_model: str = None):
        self.embed_model = embed_model or config.embed_model
        if OpenAI is None:
            self._client = None
        else:
            self._client = OpenAI(
                base_url=config.ollama_base_url,
                api_key="ollama",
            )

    def embed_query(self, text: str):
        if self._client is None:
            raise RuntimeError("openai is required to compute embeddings.")
        response = self._client.embeddings.create(
            model=self.embed_model,
            input=[text],
        )
        return response.data[0].embedding

    def embed_documents(self, texts):
        if self._client is None:
            raise RuntimeError("openai is required to compute embeddings.")
        response = self._client.embeddings.create(
            model=self.embed_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def aembed_query(self, text: str):
        return self.embed_query(text)

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)


def _slugify_model_name(model: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()
    return slug or "model"


def _coerce_corpus_records(records_or_texts: Sequence[Any]) -> List[Dict[str, Any]]:
    corpus: List[Dict[str, Any]] = []
    for index, item in enumerate(records_or_texts):
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            corpus.append(
                {
                    "name": f"document_{index + 1}",
                    "source": f"document_{index + 1}",
                    "text": text,
                }
            )
            continue

        record = dict(item)
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        name = str(record.get("name") or record.get("filename") or f"document_{index + 1}")
        record["name"] = name
        record.setdefault("source", name)
        record["text"] = text
        corpus.append(record)
    return corpus


def _coerce_query_records(records: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not records:
        return []

    normalized: List[Dict[str, Any]] = []
    for item in records:
        record = dict(item)
        query = str(record.get("query") or record.get("question") or "").strip()
        if not query:
            continue

        normalized.append(
            {
                **record,
                "query": query,
                "relevant_ids": list(record.get("relevant_ids", [])),
                "relevant_sources": list(record.get("relevant_sources", [])),
                "metadata": dict(record.get("metadata", {})),
            }
        )
    return normalized


def _extract_reference(record: Dict[str, Any]) -> str:
    for key in ("reference", "reference_answer", "expected_answer", "answer"):
        value = record.get(key)
        if value:
            return str(value)
    return ""


def _average_metric(rows: pd.DataFrame, column: str) -> float:
    if column not in rows.columns or rows.empty:
        return 0.0
    return float(rows[column].mean())


@dataclass
class EmbeddingBenchmark:
    embed_model: str
    latencies: List[float] = field(default_factory=list)

    def benchmark_latency(self, texts: List[str], num_runs: int = 3) -> Dict[str, float]:
        embeddings = OllamaEmbeddings(embed_model=self.embed_model)

        times: List[float] = []
        for _ in range(num_runs):
            start = time.perf_counter()
            embeddings.embed_documents(texts)
            times.append(time.perf_counter() - start)

        self.latencies = times
        return {
            "mean_latency": sum(times) / len(times),
            "min_latency": min(times),
            "max_latency": max(times),
            "num_texts": len(texts),
        }


@dataclass
class EmbeddingComparisonResult:
    experiment_name: str
    timestamp: str
    summary: Dict[str, float]
    per_model_results: pd.DataFrame
    per_query_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_sample_results: pd.DataFrame = field(default_factory=pd.DataFrame)


class EmbeddingComparisonBenchmark:
    def __init__(
        self,
        texts: Sequence[Any],
        models: Sequence[str],
        num_runs: int = 3,
        top_k: int = 5,
        collection_prefix: str = "embedding_comparison",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embed_model: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        self.corpus_records = _coerce_corpus_records(texts)
        self.models = [model for model in models if model]
        self.num_runs = num_runs
        self.top_k = top_k
        self.collection_prefix = collection_prefix
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_model = embed_model or config.embed_model
        self.llm_model = llm_model or config.llm_model

    def _make_dataset(self, query_records: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return _coerce_query_records(query_records)

    def _index_corpus(self, model: str, corpus_records: Sequence[Dict[str, Any]]) -> tuple[str, int, float]:
        chunker = DocumentChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        collection_name = f"{self.collection_prefix}_{_slugify_model_name(model)}_{int(time.time() * 1000)}"
        vector_store = VectorStore(collection_name=collection_name, embed_model=model)

        chunks = []
        for record in corpus_records:
            metadata = dict(record.get("metadata", {}))
            metadata.update(
                {
                    "source": record.get("source") or record.get("name"),
                    "filename": record.get("name") or record.get("filename"),
                }
            )
            chunks.extend(chunker.chunk_text(record["text"], metadata=metadata))

        start = time.perf_counter()
        vector_store.add_chunks(
            chunks,
            run_id=collection_name,
            extra_payload={
                "collection_name": collection_name,
                "embed_model": model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
        )
        indexing_latency = time.perf_counter() - start
        return collection_name, len(chunks), indexing_latency

    def _benchmark_model(
        self,
        model: str,
        query_records: Sequence[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        latency_benchmark = EmbeddingBenchmark(embed_model=model)

        document_texts = [record["text"] for record in self.corpus_records]
        query_texts = [record["query"] for record in query_records]
        document_latency = latency_benchmark.benchmark_latency(document_texts or query_texts, self.num_runs)
        query_latency = (
            latency_benchmark.benchmark_latency(query_texts, self.num_runs)
            if query_texts
            else document_latency
        )

        model_row: Dict[str, Any] = {
            "model": model,
            "num_documents": len(self.corpus_records),
            "num_queries": len(query_records),
            "num_runs": self.num_runs,
            "document_embedding_latency": document_latency["mean_latency"],
            "query_embedding_latency": query_latency["mean_latency"],
            "mean_latency": (document_latency["mean_latency"] + query_latency["mean_latency"]) / 2.0,
            "embedding_duration": (document_latency["mean_latency"] + query_latency["mean_latency"]) / 2.0,
            "latency_min": min(document_latency["min_latency"], query_latency["min_latency"]),
            "latency_max": max(document_latency["max_latency"], query_latency["max_latency"]),
            "ragas_enabled": False,
            "collection_name": "",
            "num_chunks": 0,
            "indexing_latency": 0.0,
            "indexing_duration": 0.0,
            "retrieval_duration": 0.0,
            "generation_duration": 0.0,
            "evaluation_duration": 0.0,
            "total_duration": 0.0,
            "mean_hit_rate": 0.0,
            "mean_retrieval_accuracy": 0.0,
            "mean_recall_at_k": 0.0,
            "mean_precision_at_k": 0.0,
            "mean_mrr": 0.0,
            "mean_retrieval_latency": 0.0,
            "mean_faithfulness": 0.0,
            "mean_answer_relevancy": 0.0,
            "mean_context_precision": 0.0,
            "mean_context_recall": 0.0,
        }

        per_query_frame = pd.DataFrame()
        per_sample_frame = pd.DataFrame()

        if not QDRANT_AVAILABLE or not query_records:
            return model_row, per_query_frame, per_sample_frame

        collection_name = ""
        try:
            collection_name, num_chunks, indexing_latency = self._index_corpus(model, self.corpus_records)
            pipeline = RetrievalPipeline(collection_name=collection_name, top_k=self.top_k, embed_model=model)

            retrieval_rows: List[Dict[str, Any]] = []
            ragas_queries: List[str] = []
            ragas_contexts: List[List[str]] = []
            ragas_responses: List[str] = []
            ragas_references: List[str] = []

            should_generate = getattr(pipeline, "_client", None) is not None
            for query_record in query_records:
                try:
                    result = pipeline.run(query_record["query"], include_generation=should_generate)
                except RuntimeError:
                    should_generate = False
                    result = pipeline.run(query_record["query"], include_generation=False)
                retrieval_row = evaluate_query_retrieval(
                    RetrievalQuery.from_dict(query_record),
                    result.retrieved_chunks,
                    self.top_k,
                )
                retrieval_row.update(
                    {
                        "model": model,
                        "collection_name": collection_name,
                        "embed_model": model,
                        "retrieval_latency": result.retrieval_latency,
                        "generation_latency": result.generation_latency,
                        "total_latency": result.total_latency,
                        "generated_response": result.generated_response or "",
                    }
                )
                retrieval_rows.append(retrieval_row)

                reference = _extract_reference(query_record)
                if should_generate and reference and result.generated_response:
                    ragas_queries.append(query_record["query"])
                    ragas_contexts.append([chunk.text for chunk in result.retrieved_chunks])
                    ragas_responses.append(result.generated_response)
                    ragas_references.append(reference)

            per_query_frame = pd.DataFrame(retrieval_rows)
            retrieval_summary = summarize_retrieval_metrics(per_query_frame)
            model_row.update(
                {
                    "collection_name": collection_name,
                    "num_chunks": num_chunks,
                    "indexing_latency": indexing_latency,
                    "indexing_duration": indexing_latency,
                    "mean_hit_rate": retrieval_summary.get("mean_hit_rate", 0.0),
                    "mean_retrieval_accuracy": retrieval_summary.get("mean_retrieval_accuracy", 0.0),
                    "mean_recall_at_k": retrieval_summary.get("mean_recall_at_k", 0.0),
                    "mean_precision_at_k": retrieval_summary.get("mean_precision_at_k", 0.0),
                    "mean_mrr": retrieval_summary.get("mean_mrr", 0.0),
                    "mean_retrieval_latency": retrieval_summary.get("mean_retrieval_latency", 0.0),
                    "mean_retrieval_duration": retrieval_summary.get("mean_retrieval_duration", 0.0),
                    "retrieval_duration": retrieval_summary.get("mean_retrieval_duration", 0.0),
                }
            )

            if not per_query_frame.empty:
                model_row["generation_duration"] = float(per_query_frame["generation_duration"].mean()) if "generation_duration" in per_query_frame.columns else 0.0
                model_row["retrieval_duration"] = float(per_query_frame["retrieval_duration"].mean()) if "retrieval_duration" in per_query_frame.columns else model_row["retrieval_duration"]
                model_row["total_duration"] = float(per_query_frame["total_duration"].mean()) if "total_duration" in per_query_frame.columns else (
                    model_row["indexing_duration"]
                    + model_row["retrieval_duration"]
                    + model_row["generation_duration"]
                )

            if should_generate and ragas_queries and RAGEvaluator is not None:
                evaluator = RAGEvaluator(llm_model=self.llm_model, embed_model=model)
                ragas_result = evaluator.evaluate(
                    queries=ragas_queries,
                    retrieved_contexts=ragas_contexts,
                    generated_responses=ragas_responses,
                    references=ragas_references,
                )
                per_sample_frame = ragas_result.per_sample_results.copy()
                per_sample_frame["model"] = model
                ragas_timings = getattr(ragas_result, "timings", {}) or {}
                model_row.update(
                    {
                        "ragas_enabled": True,
                        "mean_faithfulness": ragas_result.metrics.get("faithfulness", 0.0),
                        "mean_answer_relevancy": ragas_result.metrics.get("answer_relevancy", 0.0),
                        "mean_context_precision": ragas_result.metrics.get("context_precision", 0.0),
                        "mean_context_recall": ragas_result.metrics.get("context_recall", 0.0),
                        "evaluation_duration": ragas_timings.get("evaluation_duration", 0.0),
                    }
                )
                model_row["total_duration"] = model_row.get("total_duration", 0.0) + model_row.get("evaluation_duration", 0.0)
        finally:
            if collection_name:
                VectorStore(collection_name=collection_name, embed_model=model).delete_collection()

        return model_row, per_query_frame, per_sample_frame

    def run(self, query_records: Optional[Sequence[Dict[str, Any]]] = None) -> EmbeddingComparisonResult:
        if not self.models:
            raise ValueError("At least one embedding model is required.")
        if not self.corpus_records:
            raise ValueError("At least one corpus text is required.")

        normalized_queries = self._make_dataset(query_records)
        if not normalized_queries:
            normalized_queries = []

        model_rows: List[Dict[str, Any]] = []
        per_query_frames: List[pd.DataFrame] = []
        per_sample_frames: List[pd.DataFrame] = []

        for model in self.models:
            model_row, query_frame, sample_frame = self._benchmark_model(model, normalized_queries)
            model_rows.append(model_row)
            if not query_frame.empty:
                per_query_frames.append(query_frame)
            if not sample_frame.empty:
                per_sample_frames.append(sample_frame)

        per_model_results = pd.DataFrame(model_rows)
        per_query_results = pd.concat(per_query_frames, ignore_index=True) if per_query_frames else pd.DataFrame()
        per_sample_results = pd.concat(per_sample_frames, ignore_index=True) if per_sample_frames else pd.DataFrame()

        summary = {
            "num_models": float(len(per_model_results)),
            "num_documents": float(len(self.corpus_records)),
            "num_queries": float(len(normalized_queries)),
            "mean_latency": _average_metric(per_model_results, "mean_latency"),
            "mean_embedding_duration": _average_metric(per_model_results, "embedding_duration"),
            "mean_indexing_duration": _average_metric(per_model_results, "indexing_duration"),
            "mean_retrieval_duration": _average_metric(per_model_results, "retrieval_duration"),
            "mean_generation_duration": _average_metric(per_model_results, "generation_duration"),
            "mean_evaluation_duration": _average_metric(per_model_results, "evaluation_duration"),
            "mean_total_duration": _average_metric(per_model_results, "total_duration"),
            "mean_recall_at_k": _average_metric(per_model_results, "mean_recall_at_k"),
            "mean_faithfulness": _average_metric(per_model_results, "mean_faithfulness"),
            "mean_retrieval_latency": _average_metric(per_model_results, "mean_retrieval_latency"),
        }

        return EmbeddingComparisonResult(
            experiment_name="embedding_comparison",
            timestamp=datetime.now().isoformat(timespec="seconds"),
            summary=summary,
            per_model_results=per_model_results,
            per_query_results=per_query_results,
            per_sample_results=per_sample_results,
        )


def run_embedding_comparison(
    texts: Sequence[Any],
    models: Sequence[str],
    num_runs: int = 3,
    *,
    query_records: Optional[Sequence[Dict[str, Any]]] = None,
    top_k: int = 5,
    collection_prefix: str = "embedding_comparison",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embed_model: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> EmbeddingComparisonResult:
    benchmark = EmbeddingComparisonBenchmark(
        texts=texts,
        models=models,
        num_runs=num_runs,
        top_k=top_k,
        collection_prefix=collection_prefix,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model=embed_model,
        llm_model=llm_model,
    )
    return benchmark.run(query_records=query_records)


def compare_embedding_models(
    texts: Sequence[Any],
    models: Sequence[str],
    num_runs: int = 3,
    *,
    query_records: Optional[Sequence[Dict[str, Any]]] = None,
    top_k: int = 5,
    collection_prefix: str = "embedding_comparison",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embed_model: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> pd.DataFrame:
    result = run_embedding_comparison(
        texts=texts,
        models=models,
        num_runs=num_runs,
        query_records=query_records,
        top_k=top_k,
        collection_prefix=collection_prefix,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model=embed_model,
        llm_model=llm_model,
    )
    return result.per_model_results


def save_embedding_comparison(
    result: EmbeddingComparisonResult,
    filename: Optional[str] = None,
) -> Path:
    models = []
    if "model" in result.per_model_results.columns:
        models = [str(model) for model in result.per_model_results["model"].tolist()]

    artifact = BenchmarkArtifact(
        artifact_type="embedding_comparison",
        experiment_name=result.experiment_name,
        timestamp=result.timestamp,
        config={
            "num_models": int(result.summary.get("num_models", 0)),
            "num_documents": int(result.summary.get("num_documents", 0)),
            "num_queries": int(result.summary.get("num_queries", 0)),
            "models": models,
        },
        summary=result.summary,
        results=result.per_model_results,
        metadata={
            "per_query_results": result.per_query_results.to_dict(orient="records"),
            "per_sample_results": result.per_sample_results.to_dict(orient="records"),
        },
    )
    return save_benchmark_artifact(artifact, filename=filename)
