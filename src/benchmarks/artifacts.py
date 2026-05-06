from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import RESULTS_DIR


@dataclass
class BenchmarkArtifact:
    artifact_type: str
    experiment_name: str
    timestamp: str
    config: Dict[str, Any]
    summary: Dict[str, Any]
    results: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "config": self.config,
            "summary": self.summary,
            "metadata": self.metadata,
            "results_csv": self.results.to_csv(index=False),
        }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _read_json_payload(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def save_benchmark_artifact(
    artifact: BenchmarkArtifact,
    filename: Optional[str] = None,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    if filename is None:
        safe_type = artifact.artifact_type.replace(" ", "_")
        filename = f"{safe_type}_{artifact.experiment_name}_{artifact.timestamp.replace(':', '').replace('-', '')}.json"

    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / filename
    output_path.write_text(json.dumps(artifact.to_payload(), indent=2, default=_json_default))
    return output_path


def load_benchmark_artifact(path: Path) -> BenchmarkArtifact:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        timestamp = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        return BenchmarkArtifact(
            artifact_type="csv",
            experiment_name=path.stem,
            timestamp=timestamp,
            config={},
            summary={},
            results=frame,
            metadata={"path": str(path)},
        )

    payload = _read_json_payload(path)
    results_csv = payload.get("results_csv", "")
    frame = pd.read_csv(StringIO(results_csv)) if results_csv else pd.DataFrame()

    return BenchmarkArtifact(
        artifact_type=payload.get("artifact_type", "unknown"),
        experiment_name=payload.get("experiment_name", path.stem),
        timestamp=payload.get("timestamp", ""),
        config=payload.get("config", {}),
        summary=payload.get("summary", {}),
        results=frame,
        metadata=payload.get("metadata", {}),
    )


def list_benchmark_artifacts(results_dir: Path = RESULTS_DIR) -> List[Path]:
    if not results_dir.exists():
        return []

    candidates = [
        *results_dir.glob("*.json"),
        *results_dir.glob("*.csv"),
    ]
    return sorted(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, path.name.lower()),
        reverse=True,
    )
