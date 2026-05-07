import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency guard
    OpenAI = None

from ..config import config
from ..benchmarks.timing import TimingBreakdown
from .vector_store import VectorStore, RetrievedChunk

@dataclass
class RetrievalResult:
    query: str
    retrieved_chunks: List[RetrievedChunk]
    generated_response: Optional[str] = None
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0
    total_latency: float = 0.0
    timings: Dict[str, float] = field(default_factory=dict)

class RetrievalPipeline:
    def __init__(
        self,
        collection_name: str = "default",
        top_k: int = 5,
        score_threshold: float = 0.0,
        embed_model: Optional[str] = None,
    ):
        self.vector_store = VectorStore(collection_name, embed_model=embed_model or config.embed_model)
        self.top_k = top_k
        self.score_threshold = score_threshold

        if OpenAI is None:
            self._client = None
        else:
            self._client = OpenAI(
                base_url=config.ollama_base_url,
                api_key="ollama",
            )
        self.llm_model = config.llm_model

    def retrieve(self, query: str) -> tuple[List[RetrievedChunk], float]:
        start = time.perf_counter()
        results = self.vector_store.search(
            query=query,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )
        latency = time.perf_counter() - start
        return results, latency

    def generate(
        self,
        query: str,
        contexts: List[str],
        system_prompt: Optional[str] = None,
    ) -> str:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant. Use the provided context to answer the question."

        if self._client is None:
            raise RuntimeError("openai is required for generation in RetrievalPipeline.")

        context_text = "\n\n".join(f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"},
        ]

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.0,
        )
        latency = time.perf_counter() - start

        return response.choices[0].message.content, latency

    def run(self, query: str, include_generation: bool = True) -> RetrievalResult:
        start_total = time.perf_counter()

        retrieved, retrieval_latency = self.retrieve(query)

        generated_response = None
        generation_latency = 0.0

        if include_generation and retrieved:
            contexts = [chunk.text for chunk in retrieved]
            generated_response, generation_latency = self.generate(query, contexts)

        total_latency = time.perf_counter() - start_total
        timings = TimingBreakdown(
            retrieval_duration=retrieval_latency,
            generation_duration=generation_latency,
            total_duration=total_latency,
        ).to_dict()

        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved,
            generated_response=generated_response,
            retrieval_latency=retrieval_latency,
            generation_latency=generation_latency,
            total_latency=total_latency,
            timings=timings,
        )

    def run_batch(
        self,
        queries: List[str],
        include_generation: bool = True,
    ) -> List[RetrievalResult]:
        results = []
        for query in queries:
            result = self.run(query, include_generation)
            results.append(result)
        return results

def create_test_queries() -> List[Dict[str, Any]]:
    return [
        {"question": "What is Docker used for?", "expected_answer": "Docker is used for containerizing applications"},
        {"question": "What is Kubernetes?", "expected_answer": "Kubernetes is for container orchestration"},
        {"question": "What is SQL injection?", "expected_answer": "SQL injection is a vulnerability in SQL queries"},
        {"question": "What is containerization?", "expected_answer": "Containerization packages applications with their dependencies"},
        {"question": "What is CI/CD?", "expected_answer": "CI/CD is continuous integration and deployment"},
    ]
