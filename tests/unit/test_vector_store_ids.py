from types import SimpleNamespace

from src.retrieval.vector_store import VectorStore


def test_build_point_id_is_deterministic():
    chunk = SimpleNamespace(index=1, start_char=10, end_char=42)
    metadata = {"source": "/tmp/doc.md", "run_id": "run-1"}

    first = VectorStore._build_point_id("demo", chunk, "hello world", metadata)
    second = VectorStore._build_point_id("demo", chunk, "hello world", metadata)

    assert first == second
