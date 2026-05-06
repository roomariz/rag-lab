import time
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from ..config import config, RESULTS_DIR
from ..retrieval import RetrievalPipeline, VectorStore
from ..ingestion import DocumentChunker
from .datasets import RetrievalBenchmarkDataset
from .retrieval_metrics import evaluate_retrieval_benchmark


@dataclass
class ExperimentResult:
    experiment_name: str
    config: Dict[str, Any]
    results: pd.DataFrame
    summary: Dict[str, Any]


class RetrievalExperiment:
    def __init__(self, collection_name: str = "experiment"):
        self.collection_name = collection_name

    def run_top_k_tuning(
        self,
        test_queries: List[Dict[str, Any]],
        k_values: List[int] = [1, 2, 3, 5, 10],
    ) -> pd.DataFrame:
        results = []

        for k in k_values:
            print(f"Testing top_k={k}")
            pipeline = RetrievalPipeline(
                collection_name=self.collection_name,
                top_k=k,
            )

            for query_data in test_queries:
                query = query_data["question"]
                result = pipeline.run(query, include_generation=False)

                retrieved_texts = [c.text for c in result.retrieved_chunks]

                results.append({
                    "top_k": k,
                    "query": query,
                    "num_retrieved": len(result.retrieved_chunks),
                    "retrieval_latency": result.retrieval_latency,
                    "retrieved_chunks": retrieved_texts,
                })

        return pd.DataFrame(results)

    def run_labeled_retrieval_benchmark(
        self,
        dataset: RetrievalBenchmarkDataset,
        k_values: List[int] = [1, 3, 5, 10],
        embed_model: Optional[str] = None,
    ) -> pd.DataFrame:
        results = []

        for k in k_values:
            print(f"Testing labeled retrieval benchmark top_k={k}")
            pipeline = RetrievalPipeline(
                collection_name=self.collection_name,
                top_k=k,
                embed_model=embed_model,
            )
            benchmark = evaluate_retrieval_benchmark(dataset, pipeline)
            frame = benchmark.per_query_results.copy()
            frame["experiment_name"] = benchmark.experiment_name
            frame["dataset_name"] = benchmark.dataset_name
            frame["top_k"] = k
            frame["summary_hit_rate"] = benchmark.summary.get("mean_hit_rate", 0.0)
            frame["summary_recall_at_k"] = benchmark.summary.get("mean_recall_at_k", 0.0)
            frame["summary_precision_at_k"] = benchmark.summary.get("mean_precision_at_k", 0.0)
            frame["summary_mrr"] = benchmark.summary.get("mean_mrr", 0.0)
            frame["summary_retrieval_accuracy"] = benchmark.summary.get("mean_retrieval_accuracy", 0.0)
            frame["summary_retrieval_latency"] = benchmark.summary.get("mean_retrieval_latency", 0.0)
            results.append(frame)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def run_chunk_overlap_tuning(
        self,
        documents: List[str],
        test_queries: List[str],
        overlap_values: List[int] = [0, 25, 50, 100, 200],
        chunk_size: int = 512,
    ) -> pd.DataFrame:
        results = []

        for overlap in overlap_values:
            print(f"Testing chunk_overlap={overlap}")
            chunker = DocumentChunker(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
            )

            collection_name = f"overlap_{overlap}"
            vector_store = VectorStore(collection_name)

            all_chunks = []
            for doc in documents:
                chunks = chunker.chunk_text(doc)
                all_chunks.extend(chunks)

            vector_store.add_chunks(all_chunks)

            pipeline = RetrievalPipeline(
                collection_name=collection_name,
                top_k=5,
            )

            for query in test_queries:
                result = pipeline.run(query, include_generation=False)

                results.append({
                    "chunk_overlap": overlap,
                    "query": query,
                    "num_chunks": len(all_chunks),
                    "num_retrieved": len(result.retrieved_chunks),
                    "retrieval_latency": result.retrieval_latency,
                })

            vector_store.delete_collection()

        return pd.DataFrame(results)


def run_hybrid_retrieval_benchmark(
    semantic_weight: float = 0.5,
    top_k: int = 5,
) -> Dict[str, Any]:
    return {
        "semantic_weight": semantic_weight,
        "top_k": top_k,
        "status": "Not implemented - requires keyword index",
    }


def save_experiment(
    result: ExperimentResult,
    filename: Optional[str] = None,
) -> Path:
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.experiment_name}_{timestamp}.json"

    output_path = RESULTS_DIR / filename

    data = {
        "experiment_name": result.experiment_name,
        "config": result.config,
        "summary": result.summary,
        "results_csv": result.results.to_csv(index=False),
    }

    output_path.write_text(json.dumps(data, indent=2))
    return output_path
