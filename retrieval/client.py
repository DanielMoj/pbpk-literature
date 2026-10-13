"""
Qdrant client factory shared by retrieval, ingestion and the MCP tools.

Normally every call gets a client for host and port from config/settings.yaml.
Tests and the demo on the synthetic corpus register an in-memory client
(QdrantClient(":memory:")) with use_client(); all functions then use it.
"""
from qdrant_client import QdrantClient

_override: QdrantClient | None = None


def use_client(client: QdrantClient | None) -> None:
    """Route all Qdrant access to this client; None restores host/port."""
    global _override
    _override = client


def get_client(host: str, port: int) -> QdrantClient:
    """Client for host and port, or the one registered with use_client()."""
    if _override is not None:
        return _override
    return QdrantClient(host=host, port=port)
