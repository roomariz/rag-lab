import time
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency guard
    OpenAI = None

try:
    from ragas import evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas import EvaluationDataset
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._context_recall import ContextRecall
    from ragas.llms import llm_factory
    from ragas.embeddings.base import BaseRagasEmbeddings
except ImportError:  # pragma: no cover - optional dependency guard
    evaluate = None
    SingleTurnSample = None
    EvaluationDataset = None
    Faithfulness = None
    ResponseRelevancy = None
    ContextPrecision = None
    ContextRecall = None
    llm_factory = None

    class BaseRagasEmbeddings:  # type: ignore[no-redef]
        pass

from ..config import config
from .timing import TimingBreakdown


class OllamaEmbeddings(BaseRagasEmbeddings):
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


def _aggregate_ragas_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    aggregated: Dict[str, float] = {}
    for metric in metric_names:
        if metric in frame.columns and not frame[metric].empty:
            aggregated[metric] = float(frame[metric].mean())
        else:
            aggregated[metric] = 0.0
    return aggregated


@dataclass
class BenchmarkResult:
    experiment_name: str
    timestamp: str
    metrics: Dict[str, float]
    latencies: Dict[str, float]
    per_sample_results: pd.DataFrame
    timings: Dict[str, float] = field(default_factory=dict)


class RAGEvaluator:
    def __init__(self, llm_model: Optional[str] = None, embed_model: Optional[str] = None):
        self.llm_model = llm_model or config.llm_model
        self.embed_model = embed_model or config.embed_model

        if OpenAI is None:
            self._client = None
        else:
            self._client = OpenAI(
                base_url=config.ollama_base_url,
                api_key="ollama",
            )

        self.llm = (
            llm_factory(self.llm_model, client=self._client)
            if self._client is not None and llm_factory is not None
            else None
        )
        self.embeddings = OllamaEmbeddings(embed_model=self.embed_model)

        self.metrics = []
        if self.llm is not None and all(metric is not None for metric in [Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall]):
            self.metrics = [
                Faithfulness(llm=self.llm),
                ResponseRelevancy(llm=self.llm, embeddings=self.embeddings),
                ContextPrecision(llm=self.llm),
                ContextRecall(llm=self.llm),
            ]

    def evaluate(
        self,
        queries: List[str],
        retrieved_contexts: List[List[str]],
        generated_responses: List[str],
        references: List[str],
    ) -> BenchmarkResult:
        from datetime import datetime

        if self.llm is None or evaluate is None or SingleTurnSample is None or EvaluationDataset is None:
            raise RuntimeError("openai is required to run RAGAS evaluation.")

        samples = []
        for query, contexts, response, reference in zip(
            queries, retrieved_contexts, generated_responses, references
        ):
            sample = SingleTurnSample(
                user_input=query,
                retrieved_contexts=contexts,
                response=response,
                reference=reference,
            )
            samples.append(sample)

        dataset = EvaluationDataset(samples=samples)

        start_time = time.perf_counter()
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            show_progress=True,
        )
        eval_time = time.perf_counter() - start_time

        df = result.to_pandas()

        agg_metrics = _aggregate_ragas_metrics(df)

        return BenchmarkResult(
            experiment_name=f"eval_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            metrics=agg_metrics,
            latencies={"evaluation_time": eval_time, "evaluation_duration": eval_time},
            per_sample_results=df,
            timings=TimingBreakdown(evaluation_duration=eval_time, total_duration=eval_time).to_dict(),
        )


@dataclass
class EmbeddingBenchmark:
    embed_model: str
    latencies: List[float] = field(default_factory=list)

    def benchmark_latency(self, texts: List[str], num_runs: int = 3) -> Dict[str, float]:
        embeddings = OllamaEmbeddings(embed_model=self.embed_model)

        times = []
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


def compare_embedding_models(
    texts: List[str],
    models: List[str],
    num_runs: int = 3,
) -> pd.DataFrame:
    results = []

    for model in models:
        print(f"Benchmarking {model}...")
        bench = EmbeddingBenchmark(embed_model=model)
        latencies = bench.benchmark_latency(texts, num_runs)
        results.append({
            "model": model,
            **latencies,
        })

    return pd.DataFrame(results)
