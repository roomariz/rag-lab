from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class RetrievalQuery:
    query: str
    relevant_ids: List[str] = field(default_factory=list)
    relevant_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalQuery":
        return cls(
            query=data["query"],
            relevant_ids=list(data.get("relevant_ids", [])),
            relevant_sources=list(data.get("relevant_sources", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class RetrievalBenchmarkDataset:
    name: str
    queries: List[RetrievalQuery]
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalBenchmarkDataset":
        return cls(
            name=data.get("name", "retrieval_benchmark"),
            description=data.get("description", ""),
            metadata=dict(data.get("metadata", {})),
            queries=[RetrievalQuery.from_dict(item) for item in data.get("queries", [])],
        )

    @classmethod
    def from_records(
        cls,
        records: Sequence[Dict[str, Any]],
        name: str = "retrieval_benchmark",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RetrievalBenchmarkDataset":
        return cls(
            name=name,
            description=description,
            metadata=dict(metadata or {}),
            queries=[RetrievalQuery.from_dict(record) for record in records],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "queries": [asdict(query) for query in self.queries],
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=True))
        return path

    @classmethod
    def load(cls, path: Path) -> "RetrievalBenchmarkDataset":
        return cls.from_dict(json.loads(path.read_text()))


def load_query_records(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "queries" in data:
        return list(data["queries"])
    if isinstance(data, list):
        return list(data)
    raise ValueError("Unsupported retrieval dataset format.")
