from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

try:
    from qdrant_client import QdrantClient, models
    QDRANT_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency guard
    QdrantClient = None
    models = None
    QDRANT_AVAILABLE = False

from ..config import config

if TYPE_CHECKING:
    from ..ingestion.chunker import Chunk


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict
    index: int
    point_id: Optional[str] = None


class VectorStore:
    def __init__(
        self,
        collection_name: str = "default",
        embed_model: Optional[str] = None,
        vector_size: Optional[int] = None,
    ):
        self.collection_name = collection_name
        self.embed_model = embed_model or config.embed_model
        self.vector_size = vector_size
        if not QDRANT_AVAILABLE:
            raise RuntimeError(
                "qdrant-client is required for the native vector store backend."
            )
        self._client = QdrantClient(
            host=config.qdrant_host,
            port=config.qdrant_port,
        )
        self._embeddings = None

    def _get_embeddings(self):
        if self._embeddings is None:
            from ..benchmarks.evaluator import OllamaEmbeddings

            self._embeddings = OllamaEmbeddings(embed_model=self.embed_model)
        return self._embeddings

    def _collection_exists(self) -> bool:
        try:
            self._client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def collection_exists(self) -> bool:
        return self._collection_exists()

    def _ensure_collection(self, vector_size: int):
        if self._collection_exists():
            return

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    @staticmethod
    def _normalize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not metadata:
            return {}

        normalized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, os.PathLike):
                normalized[key] = os.fspath(value)
            else:
                normalized[key] = value
        return normalized

    @staticmethod
    def _build_point_id(
        collection_name: str,
        chunk: Chunk,
        text: str,
        metadata: Dict[str, Any],
    ) -> str:
        source = str(metadata.get("source", metadata.get("filename", "")))
        run_id = str(metadata.get("run_id", ""))
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        raw = "|".join(
            [
                collection_name,
                run_id,
                source,
                str(chunk.index),
                str(chunk.start_char),
                str(chunk.end_char),
                content_hash,
            ]
        )
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))

    def add_chunks(
        self,
        chunks: List[Chunk],
        batch_size: int = 100,
        run_id: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not chunks:
            return {"indexed": 0, "vector_size": self.vector_size or 0}

        texts = [chunk.text for chunk in chunks]
        embeddings = self._get_embeddings().embed_documents(texts)
        if not embeddings:
            return {"indexed": 0, "vector_size": self.vector_size or 0}

        self.vector_size = self.vector_size or len(embeddings[0])
        self._ensure_collection(self.vector_size)

        indexed = 0
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]
            points = []

            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                metadata = self._normalize_metadata(chunk.metadata)
                if run_id:
                    metadata["run_id"] = run_id
                if extra_payload:
                    metadata.update(self._normalize_metadata(extra_payload))

                metadata.setdefault("chunk_index", chunk.index)
                metadata.setdefault("start_char", chunk.start_char)
                metadata.setdefault("end_char", chunk.end_char)
                metadata.setdefault("text_length", len(chunk.text))
                metadata.setdefault("embed_model", self.embed_model)
                metadata.setdefault("content_hash", hashlib.sha256(chunk.text.encode("utf-8")).hexdigest())

                payload = {
                    "text": chunk.text,
                    "chunk_index": chunk.index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "text_length": len(chunk.text),
                    "embed_model": self.embed_model,
                    "metadata": metadata,
                }
                if run_id:
                    payload["run_id"] = run_id
                if extra_payload:
                    payload.update(self._normalize_metadata(extra_payload))

                points.append(
                    models.PointStruct(
                        id=self._build_point_id(self.collection_name, chunk, chunk.text, metadata),
                        vector=embedding,
                        payload=payload,
                    )
                )

            if points:
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                indexed += len(points)

        return {"indexed": indexed, "vector_size": self.vector_size}

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        query_filter: Optional[models.Filter] = None,
    ) -> List[RetrievedChunk]:
        if not self._collection_exists():
            raise RuntimeError(
                f"Collection '{self.collection_name}' does not exist. "
                "Index documents before searching."
            )

        embedding = self._get_embeddings().embed_query(query)
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold if score_threshold > 0 else None,
        )

        retrieved: List[RetrievedChunk] = []
        for i, result in enumerate(results):
            payload = getattr(result, "payload", None) or {}
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"metadata": metadata}

            text = payload.get("text", "")
            if not text and isinstance(metadata, dict):
                text = metadata.get("text", "")

            retrieved.append(
                RetrievedChunk(
                    text=text,
                    score=float(getattr(result, "score", 0.0)),
                    metadata=metadata,
                    index=int(payload.get("chunk_index", i)),
                    point_id=str(getattr(result, "id", "")) or None,
                )
            )

        return retrieved

    def delete_collection(self):
        try:
            self._client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass

    def get_collection_info(self) -> Dict[str, Any]:
        collection = self._client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": getattr(collection, "points_count", None),
            "indexed_vectors_count": getattr(collection, "indexed_vectors_count", None),
            "vector_size": self.vector_size,
            "embed_model": self.embed_model,
        }
