import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..config import config
from ..benchmarks.timing import TimingBreakdown
from ..retrieval.vector_store import VectorStore
from .chunker import Chunk, DocumentChunker
from .manifest import IngestionManifest, SourceFileRecord

_IGNORED_PATH_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
}


@dataclass
class IngestionResult:
    manifest: IngestionManifest
    manifest_path: Path
    collection_info: Dict[str, Any]
    source_files: List[Path]
    chunks: List[Chunk]
    indexed_chunks: int
    timings: Dict[str, float]


def _now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _normalize_sources(sources: Sequence[str | os.PathLike[str]] | str | os.PathLike[str]) -> List[Path]:
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]

    files: List[Path] = []
    for source in sources:
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            for root, dirnames, filenames in os.walk(path):
                root_path = Path(root)
                dirnames[:] = [
                    dirname
                    for dirname in sorted(dirnames)
                    if dirname not in _IGNORED_PATH_PARTS and not dirname.startswith(".")
                ]

                if any(
                    part in _IGNORED_PATH_PARTS or part.startswith(".")
                    for part in root_path.parts
                ):
                    continue

                for filename in sorted(filenames):
                    candidate = root_path / filename
                    if candidate.suffix.lower() not in config.ingestion.supported_extensions:
                        continue
                    if any(part.endswith(".egg-info") for part in candidate.parts):
                        continue
                    if any(part in _IGNORED_PATH_PARTS or part.startswith(".") for part in candidate.parts):
                        continue
                    files.append(candidate.resolve())
        elif path.is_file():
            if any(part.endswith(".egg-info") for part in path.parts):
                continue
            if any(part in _IGNORED_PATH_PARTS or part.startswith(".") for part in path.parts):
                continue
            files.append(path)
        else:
            raise FileNotFoundError(f"Source path does not exist: {path}")

    seen = set()
    ordered: List[Path] = []
    for file_path in files:
        key = str(file_path)
        if key not in seen:
            seen.add(key)
            ordered.append(file_path)

    return sorted(ordered, key=lambda item: str(item))


def preview_supported_sources(sources: Sequence[str | os.PathLike[str]] | str | os.PathLike[str]) -> List[Path]:
    return _normalize_sources(sources)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chunk_payload_metadata(
    *,
    source_path: Path,
    file_hash: str,
    chunk: Chunk,
    run_id: str,
    embed_model: str,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str],
) -> Dict[str, Any]:
    metadata = dict(chunk.metadata)
    metadata.update(
        {
            "run_id": run_id,
            "source": str(source_path),
            "filename": source_path.name,
            "file_hash": file_hash,
            "collection_name": collection_name,
            "embed_model": embed_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "separators": separators,
            "chunk_index": chunk.index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "text_length": len(chunk.text),
            "content_hash": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
        }
    )
    return metadata


def ingest_corpus(
    sources: Sequence[str | os.PathLike[str]] | str | os.PathLike[str],
    *,
    collection_name: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    separators: Optional[List[str]] = None,
    embed_model: Optional[str] = None,
    run_id: Optional[str] = None,
    manifest_dir: Optional[Path] = None,
    recreate_collection: bool = False,
) -> IngestionResult:
    source_files = _normalize_sources(sources)
    if not source_files:
        raise ValueError("No supported source files found to ingest.")

    chunk_size = chunk_size or config.chunk.chunk_size
    chunk_overlap = chunk_overlap or config.chunk.chunk_overlap
    separators = separators or ["\n\n", "\n", " ", ".", ",", "\u200b"]
    embed_model = embed_model or config.embed_model
    run_id = run_id or _now_run_id()
    manifest_dir = manifest_dir or config.ingestion.manifest_dir

    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    vector_store = VectorStore(
        collection_name=collection_name,
        embed_model=embed_model,
    )
    if recreate_collection:
        vector_store.delete_collection()

    all_chunks: List[Chunk] = []
    source_records: List[SourceFileRecord] = []
    chunking_duration = 0.0

    for source_path in source_files:
        file_hash = _file_hash(source_path)
        chunk_start = time.perf_counter()
        chunks = chunker.chunk_file(
            str(source_path),
            metadata={
                "source": str(source_path),
                "filename": source_path.name,
                "file_hash": file_hash,
                "run_id": run_id,
                "collection_name": collection_name,
                "embed_model": embed_model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "separators": separators,
            },
        )
        chunking_duration += time.perf_counter() - chunk_start

        enriched_chunks: List[Chunk] = []
        for chunk in chunks:
            metadata = _chunk_payload_metadata(
                source_path=source_path,
                file_hash=file_hash,
                chunk=chunk,
                run_id=run_id,
                embed_model=embed_model,
                collection_name=collection_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
            )
            enriched_chunks.append(
                Chunk(
                    text=chunk.text,
                    index=chunk.index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata=metadata,
                )
            )

        all_chunks.extend(enriched_chunks)
        source_records.append(
            SourceFileRecord(
                path=str(source_path),
                filename=source_path.name,
                file_hash=file_hash,
                num_chars=sum(len(chunk.text) for chunk in enriched_chunks),
                num_bytes=source_path.stat().st_size,
                metadata={
                    "chunk_count": len(enriched_chunks),
                },
            )
        )

    index_start = time.perf_counter()
    index_result = vector_store.add_chunks(
        all_chunks,
        run_id=run_id,
        extra_payload={
            "collection_name": collection_name,
            "embed_model": embed_model,
        },
    )
    vector_timings = TimingBreakdown.from_mapping(index_result.get("timings", {}))
    indexing_duration = vector_timings.indexing_duration or (time.perf_counter() - index_start)
    embedding_duration = vector_timings.embedding_duration

    manifest = IngestionManifest(
        run_id=run_id,
        created_at=IngestionManifest.create_timestamp(),
        collection_name=collection_name,
        embed_model=embed_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        source_paths=[str(path) for path in source_files],
        source_files=source_records,
        total_documents=len(source_files),
        total_chunks=len(all_chunks),
        vector_size=index_result.get("vector_size"),
        qdrant_host=config.qdrant_host,
        qdrant_port=config.qdrant_port,
        extra_metadata={
            "indexed_chunks": index_result.get("indexed", len(all_chunks)),
            "recreate_collection": recreate_collection,
            "timings": TimingBreakdown(
                chunking_duration=chunking_duration,
                embedding_duration=embedding_duration,
                indexing_duration=indexing_duration,
                total_duration=chunking_duration + vector_timings.total_duration,
            ).to_dict(),
        },
    )

    manifest_path = manifest.save(manifest_dir / f"{run_id}.json")

    return IngestionResult(
        manifest=manifest,
        manifest_path=manifest_path,
        collection_info=vector_store.get_collection_info(),
        source_files=source_files,
        chunks=all_chunks,
        indexed_chunks=index_result.get("indexed", len(all_chunks)),
        timings=TimingBreakdown(
            chunking_duration=chunking_duration,
            embedding_duration=embedding_duration,
            indexing_duration=indexing_duration,
            total_duration=chunking_duration + vector_timings.total_duration,
        ).to_dict(),
    )
