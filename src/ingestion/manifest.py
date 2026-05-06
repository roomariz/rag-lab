import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SourceFileRecord:
    path: str
    filename: str
    file_hash: str
    num_chars: int
    num_bytes: int
    status: str = "processed"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionManifest:
    run_id: str
    created_at: str
    collection_name: str
    embed_model: str
    chunk_size: int
    chunk_overlap: int
    separators: List[str]
    source_paths: List[str]
    source_files: List[SourceFileRecord]
    total_documents: int
    total_chunks: int
    vector_size: Optional[int] = None
    qdrant_host: Optional[str] = None
    qdrant_port: Optional[int] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["source_files"] = [asdict(record) for record in self.source_files]
        return data

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=True))
        return path

    @classmethod
    def load(cls, path: Path) -> "IngestionManifest":
        raw = json.loads(path.read_text())
        raw["source_files"] = [SourceFileRecord(**item) for item in raw.get("source_files", [])]
        return cls(**raw)

    @classmethod
    def create_timestamp(cls) -> str:
        return datetime.now(timezone.utc).isoformat()
