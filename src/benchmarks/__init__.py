from .datasets import RetrievalQuery, RetrievalBenchmarkDataset, load_query_records
from .retrieval_metrics import (
    RetrievalBenchmarkResult,
    evaluate_query_retrieval,
    evaluate_retrieval_benchmark,
    summarize_retrieval_metrics,
)
from .experiments import RetrievalExperiment, ExperimentResult, save_experiment
from .chunking_experiments import (
    ChunkingStrategySpec,
    ChunkingBenchmarkResult,
    run_chunking_quality_benchmark,
    save_chunking_benchmark,
)
from .artifacts import BenchmarkArtifact, load_benchmark_artifact, list_benchmark_artifacts, save_benchmark_artifact

try:
    from .evaluator import RAGEvaluator, BenchmarkResult, compare_embedding_models, EmbeddingBenchmark
except ImportError:  # pragma: no cover - optional evaluation dependencies
    RAGEvaluator = None
    BenchmarkResult = None
    compare_embedding_models = None
    EmbeddingBenchmark = None

__all__ = [
    "RAGEvaluator",
    "BenchmarkResult",
    "compare_embedding_models",
    "EmbeddingBenchmark",
    "RetrievalQuery",
    "RetrievalBenchmarkDataset",
    "load_query_records",
    "RetrievalBenchmarkResult",
    "evaluate_query_retrieval",
    "evaluate_retrieval_benchmark",
    "summarize_retrieval_metrics",
    "RetrievalExperiment",
    "ExperimentResult",
    "save_experiment",
    "ChunkingStrategySpec",
    "ChunkingBenchmarkResult",
    "run_chunking_quality_benchmark",
    "save_chunking_benchmark",
    "BenchmarkArtifact",
    "save_benchmark_artifact",
    "load_benchmark_artifact",
    "list_benchmark_artifacts",
]
