from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional


TIMING_FIELDS = (
    "chunking_duration",
    "embedding_duration",
    "indexing_duration",
    "retrieval_duration",
    "generation_duration",
    "evaluation_duration",
    "total_duration",
)

TIMING_ALIASES = {
    "chunking_latency": "chunking_duration",
    "embedding_latency": "embedding_duration",
    "indexing_latency": "indexing_duration",
    "retrieval_latency": "retrieval_duration",
    "generation_latency": "generation_duration",
    "evaluation_time": "evaluation_duration",
    "evaluation_latency": "evaluation_duration",
    "total_latency": "total_duration",
}


@dataclass
class TimingBreakdown:
    chunking_duration: float = 0.0
    embedding_duration: float = 0.0
    indexing_duration: float = 0.0
    retrieval_duration: float = 0.0
    generation_duration: float = 0.0
    evaluation_duration: float = 0.0
    total_duration: float = 0.0

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]] = None, **overrides: Any) -> "TimingBreakdown":
        values: Dict[str, float] = {field: 0.0 for field in TIMING_FIELDS}

        def apply_mapping(mapping: Mapping[str, Any]) -> None:
            for key, value in mapping.items():
                canonical_key = TIMING_ALIASES.get(key, key)
                if canonical_key in values and value is not None:
                    try:
                        values[canonical_key] = float(value)
                    except (TypeError, ValueError):
                        continue

        if payload:
            if is_dataclass(payload):
                apply_mapping(asdict(payload))
            elif hasattr(payload, "to_dict") and not isinstance(payload, Mapping):
                apply_mapping(payload.to_dict())  # type: ignore[call-arg]
            else:
                apply_mapping(payload)
        if overrides:
            apply_mapping(overrides)

        if not values["total_duration"]:
            total = sum(
                values[field]
                for field in TIMING_FIELDS
                if field != "total_duration" and values[field] > 0
            )
            if total:
                values["total_duration"] = total

        return cls(**values)

    def to_dict(self) -> Dict[str, float]:
        return {field: float(getattr(self, field, 0.0)) for field in TIMING_FIELDS}

    def to_nonzero_dict(self) -> Dict[str, float]:
        return {key: value for key, value in self.to_dict().items() if value > 0}


def normalize_timing_payload(payload: Optional[Mapping[str, Any]] = None, **overrides: Any) -> Dict[str, float]:
    return TimingBreakdown.from_mapping(payload, **overrides).to_dict()


def attach_timing_columns(
    frame,
    timing: Optional[Mapping[str, Any]] = None,
    *,
    prefix: str = "",
):
    timing_breakdown = TimingBreakdown.from_mapping(timing)
    for field, value in timing_breakdown.to_dict().items():
        frame[f"{prefix}{field}"] = value
    return frame
