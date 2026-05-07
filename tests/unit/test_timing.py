import pandas as pd

from src.benchmarks.timing import (
    TIMING_FIELDS,
    TimingBreakdown,
    attach_timing_columns,
    normalize_timing_payload,
)


def test_normalize_timing_payload_maps_aliases_and_fills_total():
    payload = normalize_timing_payload(
        {
            "retrieval_latency": 0.2,
            "generation_latency": 0.3,
            "evaluation_time": 0.5,
        }
    )

    assert payload["retrieval_duration"] == 0.2
    assert payload["generation_duration"] == 0.3
    assert payload["evaluation_duration"] == 0.5
    assert payload["total_duration"] == 1.0
    assert all(field in payload for field in TIMING_FIELDS)


def test_attach_timing_columns_adds_standard_schema():
    frame = pd.DataFrame([{"model": "demo"}])
    result = attach_timing_columns(frame, TimingBreakdown(chunking_duration=0.1, indexing_duration=0.2))

    assert "chunking_duration" in result.columns
    assert "indexing_duration" in result.columns
    assert result.loc[0, "chunking_duration"] == 0.1
    assert result.loc[0, "indexing_duration"] == 0.2
