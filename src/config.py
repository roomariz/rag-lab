import os
from dataclasses import dataclass, field
from pathlib import Path

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

LLM_MODEL = os.getenv("LLM_MODEL", "mistral:latest")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "qdrant")

@dataclass
class ChunkConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50

@dataclass
class RetrievalConfig:
    top_k: int = 5
    score_threshold: float = 0.0


@dataclass
class IngestionConfig:
    manifest_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "INGESTION_MANIFEST_DIR",
                str(Path(__file__).parent.parent / "results" / "ingestions"),
            )
        )
    )
    supported_extensions: tuple[str, ...] = (".txt", ".md", ".pdf", ".docx")


@dataclass
class Config:
    ollama_base_url: str = OLLAMA_BASE_URL
    llm_model: str = LLM_MODEL
    embed_model: str = EMBED_MODEL
    qdrant_host: str = QDRANT_HOST
    qdrant_port: int = QDRANT_PORT
    vector_store_backend: str = VECTOR_STORE_BACKEND
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)

    @property
    def embed_model_name(self) -> str:
        return self.embed_model.replace(":latest", "").replace(":", "-")

config = Config()

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

INGESTION_DIR = RESULTS_DIR / "ingestions"
INGESTION_DIR.mkdir(exist_ok=True)
