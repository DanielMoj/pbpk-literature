"""
Embeds chunks via bge-large-en-v1.5 and stores in Qdrant.
Dense vectors (1024-dim) + BM25 sparse vectors for hybrid search.
Idempotent: uses DOI + section_index as dedup key.

Input:  Chunks from chunk_papers.py
Output: Qdrant collection 'papers' populated
"""
import hashlib
import logging
from itertools import islice

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct,
    Filter,
    FilterSelector,
    FieldCondition,
    MatchValue,
    SparseVector,
)
from retrieval.client import get_client
from retrieval.models import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_id_to_point_id(chunk_id: str) -> int:
    """Deterministic 64-bit int from chunk_id for Qdrant point ID."""
    h = hashlib.sha256(chunk_id.encode()).hexdigest()
    return int(h[:16], 16)  # 64-bit unsigned


def _batched(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def _chunk_to_payload(chunk: Chunk) -> dict:
    """Flatten chunk + metadata into Qdrant payload dict."""
    m = chunk.metadata
    a = m.annotation
    return {
        "chunk_id": chunk.chunk_id,
        "doi": chunk.doi,
        "section_type": chunk.section_type,
        "text": chunk.text,
        "title": m.title,
        "authors": m.authors,
        "journal": m.journal,
        "year": m.year,
        "abstract": m.abstract,
        "gap_status": a.gap_status.value,
        "tags": a.tags,
        "relevant_for": a.relevant_for,
        "note": a.note,
    }


# ---------------------------------------------------------------------------
# BM25 sparse vectors via fastembed
# ---------------------------------------------------------------------------

_bm25_model = None


def _get_bm25_model():
    global _bm25_model
    if _bm25_model is None:
        from fastembed import SparseTextEmbedding
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("BM25 model loaded")
    return _bm25_model


def _encode_sparse(texts: list[str]) -> list[SparseVector]:
    """Encode texts into BM25 sparse vectors."""
    try:
        model = _get_bm25_model()
        results = list(model.embed(texts))
        sparse_vectors = []
        for sparse in results:
            sparse_vectors.append(SparseVector(
                indices=sparse.indices.tolist(),
                values=sparse.values.tolist(),
            ))
        return sparse_vectors
    except Exception as e:
        logger.warning(f"BM25 encoding failed ({e}), using empty sparse vectors")
        return [SparseVector(indices=[], values=[]) for _ in texts]

# ---------------------------------------------------------------------------
# Dense embedding via SentenceTransformer
# ---------------------------------------------------------------------------

_dense_model = None


def _get_dense_model(model_name: str, device: str):
    global _dense_model
    if _dense_model is None:
        from sentence_transformers import SentenceTransformer
        from config.loader import resolve_device
        device = resolve_device(device)
        logger.info(f"Loading {model_name} on {device}...")
        _dense_model = SentenceTransformer(model_name, device=device)
        logger.info("Dense model loaded")
    return _dense_model


def _encode_dense(
    texts: list[str],
    model_name: str,
    device: str,
    batch_size: int,
) -> list[list[float]]:
    """Encode texts into dense vectors."""
    model = _get_dense_model(model_name, device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# Qdrant operations
# ---------------------------------------------------------------------------

def _delete_doi_chunks(client: QdrantClient, collection: str, doi: str):
    """Delete all existing chunks for a DOI before re-ingestion."""
    client.delete(
        collection_name=collection,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="doi", match=MatchValue(value=doi))]
            )
        ),
    )


def _upsert_batch(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[SparseVector],
):
    """Upsert a batch of chunks with their vectors."""
    points = []
    for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors):
        point_id = _chunk_id_to_point_id(chunk.chunk_id)
        points.append(PointStruct(
            id=point_id,
            vector={
                "dense": dense,
                "bm25": sparse,
            },
            payload=_chunk_to_payload(chunk),
        ))

    client.upsert(
        collection_name=collection,
        points=points,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def embed_and_store(
    chunks: list[Chunk],
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection: str = "papers",
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: str = "auto",
    batch_size: int = 64,
    delete_before_insert: bool = True,
):
    """
    Embed chunks and store in Qdrant.

    Args:
        chunks: List of Chunk objects from chunk_papers.py
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        collection: Qdrant collection name
        model_name: SentenceTransformer model for dense embeddings
        device: 'auto' (CUDA if available, else CPU), 'cuda' or 'cpu'
        batch_size: Batch size for encoding
        delete_before_insert: Delete existing chunks per DOI first (idempotency)
    """
    if not chunks:
        logger.warning("Keine Chunks zum Ingestieren")
        return

    client = get_client(qdrant_host, qdrant_port)

    # Verify collection exists
    if not client.collection_exists(collection):
        raise RuntimeError(
            f"Collection '{collection}' existiert nicht. "
            f"Zuerst 'python scripts/setup_qdrant.py' ausführen."
        )

    # Group chunks by DOI for idempotent re-ingestion
    if delete_before_insert:
        dois = {c.doi for c in chunks}
        logger.info(f"Lösche existierende Chunks für {len(dois)} DOIs...")
        for doi in dois:
            _delete_doi_chunks(client, collection, doi)

    # Encode all texts
    texts = [c.text for c in chunks]
    logger.info(f"Erzeuge Dense Embeddings für {len(texts)} Chunks...")
    dense_vectors = _encode_dense(texts, model_name, device, batch_size)

    logger.info(f"Erzeuge BM25-Vektoren für {len(texts)} Chunks...")
    sparse_vectors = _encode_sparse(texts)

    # Upsert in batches
    total_upserted = 0
    for batch_chunks, batch_dense, batch_sparse in zip(
        _batched(chunks, batch_size),
        _batched(dense_vectors, batch_size),
        _batched(sparse_vectors, batch_size),
    ):
        _upsert_batch(client, collection, batch_chunks, batch_dense, batch_sparse)
        total_upserted += len(batch_chunks)
        logger.debug(f"  Upserted {total_upserted}/{len(chunks)}")

    logger.info(f"Ingestion fertig: {total_upserted} Chunks in '{collection}'")


def embed_and_store_from_settings(chunks: list[Chunk]):
    """Convenience: read all params from config/settings.yaml."""
    from config.loader import settings
    embed_and_store(
        chunks=chunks,
        qdrant_host=settings.qdrant.host,
        qdrant_port=settings.qdrant.port,
        collection=settings.qdrant.collection,
        model_name=settings.embedding.model,
        device=settings.embedding.device,
        batch_size=settings.embedding.batch_size,
    )
