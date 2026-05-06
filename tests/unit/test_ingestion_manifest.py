from pathlib import Path

from src.ingestion.ingest import _normalize_sources
from src.ingestion.manifest import IngestionManifest, SourceFileRecord


def test_manifest_roundtrip(tmp_path: Path):
    manifest = IngestionManifest(
        run_id="run-1",
        created_at="2026-05-06T00:00:00Z",
        collection_name="demo",
        embed_model="nomic-embed-text:latest",
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n"],
        source_paths=["/tmp/doc.md"],
        source_files=[
            SourceFileRecord(
                path="/tmp/doc.md",
                filename="doc.md",
                file_hash="abc123",
                num_chars=42,
                num_bytes=128,
                metadata={"chunk_count": 3},
            )
        ],
        total_documents=1,
        total_chunks=3,
        vector_size=768,
        qdrant_host="localhost",
        qdrant_port=6333,
    )

    path = tmp_path / "manifest.json"
    saved = manifest.save(path)

    loaded = IngestionManifest.load(saved)
    assert loaded.run_id == manifest.run_id
    assert loaded.collection_name == "demo"
    assert loaded.source_files[0].filename == "doc.md"
    assert loaded.source_files[0].metadata["chunk_count"] == 3


def test_normalize_sources_discovers_supported_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.md").write_text("world")
    (tmp_path / "ignore.png").write_text("binary")

    files = _normalize_sources(tmp_path)

    assert [file_path.name for file_path in files] == ["a.txt", "b.md"]


def test_normalize_sources_skips_hidden_and_virtualenv_dirs(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "visible.md").write_text("keep me")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "Lib").mkdir()
    (tmp_path / ".venv" / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".venv" / "Lib" / "site-packages" / "api.md").write_text("ignore me")
    (tmp_path / ".hidden.md").write_text("ignore me too")

    files = _normalize_sources(tmp_path)

    assert [file_path.name for file_path in files] == ["visible.md"]


def test_normalize_sources_ignores_virtualenv_packages_tree(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "keep.md").write_text("keep me")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "Lib").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".venv" / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".venv" / "Lib" / "site-packages" / "ignore.md").write_text("ignore me")

    files = _normalize_sources(tmp_path)

    assert [file_path.name for file_path in files] == ["keep.md"]
