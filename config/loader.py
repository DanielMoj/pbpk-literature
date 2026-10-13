"""
Config loader — reads config/settings.yaml relative to project root.
"""
import yaml
from pathlib import Path
from dataclasses import dataclass, field


def _project_root() -> Path:
    """Walk up from this file until we find config/settings.yaml."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "config" / "settings.yaml").exists():
            return p
        p = p.parent
    raise FileNotFoundError("Could not locate config/settings.yaml")


PROJECT_ROOT = _project_root()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class QdrantConfig:
    host: str = "127.0.0.1"
    port: int = 6333
    collection: str = "papers"


@dataclass
class EmbeddingConfig:
    model: str = "BAAI/bge-large-en-v1.5"
    device: str = "auto"      # auto | cuda | cpu — auto: CUDA, wenn vorhanden, sonst CPU
    batch_size: int = 64


@dataclass
class ChunkingConfig:
    min_chunk_length: int = 50
    max_chunk_length: int = 350
    overlap: int = 0
    subword_factor: float = 1.3


@dataclass
class SearchConfig:
    default_top_k: int = 10
    hybrid_alpha: float = 0.7


@dataclass
class McpConfig:
    transport: str = "stdio"
    sse_port: int = 8080


@dataclass
class Settings:
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    mcp: McpConfig = field(default_factory=McpConfig)


def load_settings(path: Path | None = None) -> Settings:
    """Read settings.yaml; missing sections and keys keep their defaults."""
    raw = load_yaml(path or PROJECT_ROOT / "config" / "settings.yaml")
    return Settings(
        qdrant=QdrantConfig(**(raw.get("qdrant") or {})),
        embedding=EmbeddingConfig(**(raw.get("embedding") or {})),
        chunking=ChunkingConfig(**(raw.get("chunking") or {})),
        search=SearchConfig(**(raw.get("search") or {})),
        mcp=McpConfig(**(raw.get("mcp") or {})),
    )


def resolve_device(device: str) -> str:
    """Resolve the embedding device: 'auto' → 'cuda' if PyTorch sees a GPU, else 'cpu'.

    Other values ('cuda', 'cpu', 'cuda:1', ...) are passed through unchanged.
    """
    if device != "auto":
        return device
    import torch  # erst hier: torch ist groß und wird nur zum Laden des Modells gebraucht
    return "cuda" if torch.cuda.is_available() else "cpu"


# Singleton — importiert als `from config.loader import settings`
settings = load_settings()
