"""
Hybrid search logic:
Query → Dense Embedding + BM25 Sparse → one Qdrant query per ranking
(same metadata filter) → weighted Reciprocal Rank Fusion → Ranked Results
with snippets.

hybrid_alpha weights the two rankings: 1 = dense only, 0 = BM25 only,
in between score = alpha / (k + rank_dense) + (1 - alpha) / (k + rank_bm25)
with k = 60 as in the original RRF paper, which sums the rankings unweighted:
Cormack, Clarke und Büttcher (2009), SIGIR '09, S. 758–759, doi:10.1145/1571941.1572114

Loads the embedding models lazily on the first query (own singletons,
separate from embed_and_store.py).

Usage:
    from retrieval.search import search, search_from_settings, find_similar_from_settings

    results = search("PBPK validation statistics TOST", top_k=10)
    results = search("VPC power", tags=["misspecification"], year_min=2015)
    results = find_similar_from_settings("10.1002/psp4.12345", top_k=5)
"""
import logging
import re

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    MatchText,
    Range,
    ScoredPoint,
    SparseVector,
)

from retrieval.client import get_client
from retrieval.models import (
    Chunk,
    PaperMetadata,
    Annotation,
    GapStatus,
    SearchResult,
)

logger = logging.getLogger(__name__)

# Konstante der Reciprocal Rank Fusion, k = 60 wie bei Cormack, Clarke und Büttcher (2009)
_RRF_K = 60


# ---------------------------------------------------------------------------
# Embedding (lazy singletons, loaded on the first query)
# ---------------------------------------------------------------------------

_dense_model = None
_bm25_model = None


def _get_dense_model(model_name: str, device: str):
    global _dense_model
    if _dense_model is None:
        from sentence_transformers import SentenceTransformer
        from config.loader import resolve_device
        device = resolve_device(device)
        logger.info(f"Loading {model_name} on {device}...")
        _dense_model = SentenceTransformer(model_name, device=device)
    return _dense_model


def _get_bm25_model():
    global _bm25_model
    if _bm25_model is None:
        from fastembed import SparseTextEmbedding
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _bm25_model


def _encode_query_dense(
    query: str, model_name: str, device: str
) -> list[float]:
    """Single query → dense vector."""
    model = _get_dense_model(model_name, device)
    vec = model.encode(query, normalize_embeddings=True)
    return vec.tolist()


def _encode_query_sparse(query: str) -> SparseVector:
    """Single query → BM25 sparse vector (empty if fastembed is unavailable)."""
    try:
        model = _get_bm25_model()
        results = list(model.query_embed(query))
    except Exception as e:
        logger.warning(f"BM25 nicht verfügbar ({e}) — Suche nur über Embeddings")
        return SparseVector(indices=[], values=[])
    if not results:
        return SparseVector(indices=[], values=[])
    sparse = results[0]
    return SparseVector(
        indices=sparse.indices.tolist(),
        values=sparse.values.tolist(),
    )


# ---------------------------------------------------------------------------
# Filter builder
# ---------------------------------------------------------------------------

def build_filter(
    journal: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    authors: list[str] | None = None,
    section_type: str | None = None,
    tags: list[str] | None = None,
    gap_status: str | None = None,
    relevant_for: str | list[str] | None = None,
    doi: str | None = None,
) -> Filter | None:
    """
    Translate search parameters into a Qdrant Filter.
    Returns None if no filters are specified.
    """
    conditions = []

    if doi:
        conditions.append(
            FieldCondition(key="doi", match=MatchValue(value=doi))
        )
    if journal:
        conditions.append(
            FieldCondition(key="journal", match=MatchValue(value=journal))
        )
    if year_min is not None:
        conditions.append(
            FieldCondition(key="year", range=Range(gte=year_min))
        )
    if year_max is not None:
        conditions.append(
            FieldCondition(key="year", range=Range(lte=year_max))
        )
    if authors:
        # MatchAny: at least one author in the list
        conditions.append(
            FieldCondition(key="authors", match=MatchAny(any=authors))
        )
    if section_type:
        conditions.append(
            FieldCondition(key="section_type", match=MatchValue(value=section_type))
        )
    if tags:
        for tag in tags:
            conditions.append(
                FieldCondition(key="tags", match=MatchValue(value=tag))
            )
    if gap_status:
        conditions.append(
            FieldCondition(key="gap_status", match=MatchValue(value=gap_status))
        )
    if relevant_for:
        if isinstance(relevant_for, str):
            relevant_for = [relevant_for]
        for rf in relevant_for:
            conditions.append(
                FieldCondition(key="relevant_for", match=MatchValue(value=rf))
            )

    return Filter(must=conditions) if conditions else None


# ---------------------------------------------------------------------------
# Payload → Pydantic reconstruction
# ---------------------------------------------------------------------------

def _payload_to_chunk(payload: dict) -> Chunk:
    """Reconstruct a Chunk from Qdrant payload."""
    annotation = Annotation(
        gap_status=GapStatus(payload.get("gap_status", "UNASSESSED")),
        tags=payload.get("tags", []),
        relevant_for=payload.get("relevant_for", []),
        note=payload.get("note", ""),
    )
    metadata = PaperMetadata(
        doi=payload.get("doi", ""),
        title=payload.get("title", ""),
        authors=payload.get("authors", []),
        journal=payload.get("journal", ""),
        year=payload.get("year", 0),
        abstract=payload.get("abstract", ""),
        annotation=annotation,
    )
    return Chunk(
        chunk_id=payload.get("chunk_id", ""),
        doi=payload.get("doi", ""),
        section_type=payload.get("section_type", ""),
        text=payload.get("text", ""),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Snippet extraction
# ---------------------------------------------------------------------------

_SNIPPET_WORDS = 150  # ~200 tokens nach subword-expansion


def _extract_snippet(text: str, query: str) -> str:
    """
    Extract the most relevant ~200-token window from text.

    Strategy: find the paragraph with the highest overlap of query terms,
    then return that paragraph (trimmed if too long). Falls back to the
    first N words if no query terms match.
    """
    if not text:
        return ""

    query_terms = {t.lower() for t in query.split() if len(t) > 2}
    if not query_terms:
        words = text.split()
        return " ".join(words[:_SNIPPET_WORDS])

    # Score paragraphs by query-term overlap
    paragraphs = re.split(r"\n\s*\n", text)
    if not paragraphs:
        paragraphs = [text]

    best_para = paragraphs[0]
    best_score = -1

    for para in paragraphs:
        para_lower = para.lower()
        score = sum(1 for t in query_terms if t in para_lower)
        if score > best_score:
            best_score = score
            best_para = para

    # Trim to snippet length
    words = best_para.split()
    if len(words) > _SNIPPET_WORDS:
        return " ".join(words[:_SNIPPET_WORDS]) + " …"
    return best_para


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

def search(
    query: str,
    *,
    # Connection
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection: str = "papers",
    # Embedding
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: str = "auto",
    # Search params
    top_k: int = 10,
    hybrid_alpha: float = 0.7,
    prefetch_limit: int | None = None,
    # Metadata filters
    journal: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    authors: list[str] | None = None,
    section_type: str | None = None,
    tags: list[str] | None = None,
    gap_status: str | None = None,
    relevant_for: str | list[str] | None = None,
    doi: str | None = None,
) -> list[SearchResult]:
    """
    Hybrid search: dense (bge-large) + sparse (BM25), fused by weighted RRF.

    hybrid_alpha: weight of the dense ranking (1 = dense only, 0 = BM25 only).
    A ranking with weight 0 is not queried at all.
    prefetch_limit: candidates per ranking before fusion (default max(3 × top_k, 30)).

    If query is empty, falls back to metadata-only filter search
    (scroll order, no relevance score).
    """
    if not 0.0 <= hybrid_alpha <= 1.0:
        raise ValueError(f"hybrid_alpha muss zwischen 0 und 1 liegen, nicht {hybrid_alpha}")

    client = get_client(qdrant_host, qdrant_port)
    qfilter = build_filter(
        journal=journal, year_min=year_min, year_max=year_max,
        authors=authors, section_type=section_type, tags=tags,
        gap_status=gap_status, relevant_for=relevant_for, doi=doi,
    )

    if prefetch_limit is None:
        prefetch_limit = max(top_k * 3, 30)

    # --- Empty query: metadata-only scroll ---
    if not query.strip():
        return _metadata_only_search(
            client, collection, qfilter, top_k
        )

    # --- One ranking per vector type, same filter ---
    rankings: list[tuple[float, list[ScoredPoint]]] = []

    if hybrid_alpha > 0:
        dense_vec = _encode_query_dense(query, model_name, device)
        response = client.query_points(
            collection_name=collection,
            query=dense_vec,
            using="dense",
            limit=prefetch_limit,
            query_filter=qfilter,
            with_payload=True,
        )
        rankings.append((hybrid_alpha, response.points))

    if hybrid_alpha < 1:
        sparse_vec = _encode_query_sparse(query)
        if sparse_vec.indices:  # leer: fastembed fehlt oder nur Stoppwörter
            response = client.query_points(
                collection_name=collection,
                query=sparse_vec,
                using="bm25",
                limit=prefetch_limit,
                query_filter=qfilter,
                with_payload=True,
            )
            rankings.append((1.0 - hybrid_alpha, response.points))

    return _fused_to_results(_weighted_rrf(rankings, top_k), query)


def _weighted_rrf(
    rankings: list[tuple[float, list[ScoredPoint]]], limit: int
) -> list[tuple[float, ScoredPoint]]:
    """
    Weighted Reciprocal Rank Fusion: score = Σ weight / (k + rank), rank from 1.

    Points missing from a ranking get nothing from it. Ties keep the order of
    first appearance (dense before BM25), so the result is deterministic.
    """
    fused: dict = {}
    for weight, points in rankings:
        for rank, point in enumerate(points, start=1):
            score, first_seen = fused.get(point.id, (0.0, point))
            fused[point.id] = (score + weight / (_RRF_K + rank), first_seen)

    ranked = sorted(fused.values(), key=lambda entry: entry[0], reverse=True)
    return ranked[:limit]


def _metadata_only_search(
    client: QdrantClient,
    collection: str,
    qfilter: Filter | None,
    top_k: int,
) -> list[SearchResult]:
    """Scroll-based search when no query text is provided."""
    if qfilter is None:
        logger.warning("Leere Query ohne Filter — gebe leere Ergebnisse zurück")
        return []

    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=qfilter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    results = []
    for pt in points:
        chunk = _payload_to_chunk(pt.payload)
        snippet = _extract_snippet(chunk.text, "")
        results.append(SearchResult(
            chunk=chunk,
            score=0.0,  # no relevance score for scroll
            snippet=snippet,
        ))
    return results


def _fused_to_results(
    fused: list[tuple[float, ScoredPoint]], query: str
) -> list[SearchResult]:
    """Convert fused (score, point) pairs to List[SearchResult]."""
    results = []
    for score, point in fused:
        chunk = _payload_to_chunk(point.payload)
        snippet = _extract_snippet(chunk.text, query)
        results.append(SearchResult(
            chunk=chunk,
            score=score,
            snippet=snippet,
        ))
    return results


# ---------------------------------------------------------------------------
# find_similar: nearest neighbors from an existing document vector
# ---------------------------------------------------------------------------

def find_similar(
    doi: str,
    section: str = "abstract",
    *,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection: str = "papers",
    top_k: int = 10,
    exclude_same_doi: bool = True,
) -> list[SearchResult]:
    """
    Find papers similar to a given DOI by searching with its embedding.

    Retrieves the dense vector of the specified section and uses it
    as the query vector. BM25 is skipped since we search by vector,
    not by text.
    """
    client = get_client(qdrant_host, qdrant_port)

    # Find the anchor chunk
    anchor_filter = Filter(must=[
        FieldCondition(key="doi", match=MatchValue(value=doi)),
        FieldCondition(key="section_type", match=MatchValue(value=section)),
    ])
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=anchor_filter,
        limit=1,
        with_payload=True,
        with_vectors=["dense"],
    )

    if not points:
        # Fallback: try any section for this DOI
        fallback_filter = Filter(must=[
            FieldCondition(key="doi", match=MatchValue(value=doi)),
        ])
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=fallback_filter,
            limit=1,
            with_payload=True,
            with_vectors=["dense"],
        )
        if not points:
            logger.warning(f"DOI {doi} nicht in Collection gefunden")
            return []

    anchor_vector = points[0].vector["dense"]

    # Exclude chunks from the same paper
    exclude_filter = None
    if exclude_same_doi:
        exclude_filter = Filter(must_not=[
            FieldCondition(key="doi", match=MatchValue(value=doi))
        ])

    # Search by vector — request more to account for multiple chunks per paper
    response = client.query_points(
        collection_name=collection,
        query=anchor_vector,
        using="dense",
        limit=top_k * 3,
        query_filter=exclude_filter,
        with_payload=True,
    )

    # Deduplicate by DOI — keep highest-scoring chunk per paper
    seen_dois: dict[str, SearchResult] = {}
    for point in response.points:
        chunk = _payload_to_chunk(point.payload)
        score = point.score if point.score is not None else 0.0
        if chunk.doi not in seen_dois or score > seen_dois[chunk.doi].score:
            snippet = _extract_snippet(chunk.text, "")
            seen_dois[chunk.doi] = SearchResult(
                chunk=chunk, score=score, snippet=snippet,
            )

    ranked = sorted(seen_dois.values(), key=lambda r: r.score, reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Author search (metadata-only, no embedding)
# ---------------------------------------------------------------------------

def search_by_author(
    author: str,
    *,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection: str = "papers",
    top_k: int = 20,
) -> list[SearchResult]:
    """
    Find all papers by an author. Metadata filter, no embedding needed.
    Deduplicates by DOI, returns one result per paper.

    Uses Qdrant's text match on the authors list, so a surname such as
    'Mustermann' also finds entries like 'Erika Mustermann, Beispiel-Universität'
    (GROBID often keeps the affiliation in the author string).
    """
    client = get_client(qdrant_host, qdrant_port)

    author_filter = Filter(must=[
        FieldCondition(key="authors", match=MatchText(text=author))
    ])

    # Scroll through — authors can have many papers
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=author_filter,
        limit=top_k * 5,  # multiple chunks per paper
        with_payload=True,
        with_vectors=False,
    )

    # Deduplicate: keep abstract chunk if available, else first chunk
    papers: dict[str, SearchResult] = {}
    for pt in points:
        chunk = _payload_to_chunk(pt.payload)
        existing = papers.get(chunk.doi)
        # Prefer abstract section
        if existing is None:
            snippet = _extract_snippet(chunk.text, "")
            papers[chunk.doi] = SearchResult(
                chunk=chunk, score=0.0, snippet=snippet,
            )
        elif chunk.section_type == "abstract" and existing.chunk.section_type != "abstract":
            snippet = _extract_snippet(chunk.text, "")
            papers[chunk.doi] = SearchResult(
                chunk=chunk, score=0.0, snippet=snippet,
            )

    # Sort by year descending
    ranked = sorted(papers.values(), key=lambda r: r.chunk.metadata.year, reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Gap landscape (aggregation query)
# ---------------------------------------------------------------------------

def get_gap_landscape(
    *,
    query: str | None = None,
    relevant_for: str | None = None,
    gap_status: str | None = None,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection: str = "papers",
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: str = "auto",
    hybrid_alpha: float = 0.7,
    top_k: int = 100,
) -> dict:
    """
    Aggregated overview of gap annotations.
    Without gap_status, UNASSESSED papers are left out (with and without query).

    Returns dict with:
        - groups: {gap_status_value: [entry, ...]}
        - summary: {status: count, total_annotated: int}
    """
    # If a query is given, use hybrid search; otherwise metadata filter
    if query and query.strip():
        results = search(
            query,
            qdrant_host=qdrant_host, qdrant_port=qdrant_port,
            collection=collection, model_name=model_name, device=device,
            top_k=top_k, hybrid_alpha=hybrid_alpha,
            relevant_for=relevant_for, gap_status=gap_status,
        )
        if gap_status is None:
            results = [
                r for r in results
                if r.chunk.metadata.annotation.gap_status != GapStatus.UNASSESSED
            ]
    else:
        client = get_client(qdrant_host, qdrant_port)
        qfilter = build_filter(
            gap_status=gap_status, relevant_for=relevant_for,
        )
        # Exclude UNASSESSED unless explicitly requested
        if gap_status is None:
            must_not = [FieldCondition(
                key="gap_status", match=MatchValue(value="UNASSESSED")
            )]
            if qfilter:
                qfilter.must_not = must_not
            else:
                qfilter = Filter(must_not=must_not)

        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=qfilter,
            limit=top_k * 5,
            with_payload=True,
            with_vectors=False,
        )
        results = []
        for pt in points:
            chunk = _payload_to_chunk(pt.payload)
            results.append(SearchResult(
                chunk=chunk, score=0.0,
                snippet=_extract_snippet(chunk.text, ""),
            ))

    # Deduplicate by DOI
    seen: dict[str, SearchResult] = {}
    for r in results:
        if r.chunk.doi not in seen:
            seen[r.chunk.doi] = r

    # Group by gap_status
    groups: dict[str, list[dict]] = {}
    for r in seen.values():
        status = r.chunk.metadata.annotation.gap_status.value
        entry = {
            "doi": r.chunk.doi,
            "title": r.chunk.metadata.title,
            "year": r.chunk.metadata.year,
            "tags": r.chunk.metadata.annotation.tags,
            "relevant_for": r.chunk.metadata.annotation.relevant_for,
            "note": r.chunk.metadata.annotation.note,
        }
        groups.setdefault(status, []).append(entry)

    # Summary counts
    summary = {status: len(entries) for status, entries in groups.items()}
    summary["total_annotated"] = sum(summary.values())

    return {"groups": groups, "summary": summary}


# ---------------------------------------------------------------------------
# Convenience: settings-aware wrappers
# ---------------------------------------------------------------------------

def search_from_settings(query: str, **kwargs) -> list[SearchResult]:
    """search() with connection + model params from config/settings.yaml."""
    from config.loader import settings
    defaults = dict(
        qdrant_host=settings.qdrant.host,
        qdrant_port=settings.qdrant.port,
        collection=settings.qdrant.collection,
        model_name=settings.embedding.model,
        device=settings.embedding.device,
        top_k=settings.search.default_top_k,
        hybrid_alpha=settings.search.hybrid_alpha,
    )
    defaults.update(kwargs)
    return search(query, **defaults)


def find_similar_from_settings(doi: str, **kwargs) -> list[SearchResult]:
    """find_similar() with connection params from config/settings.yaml."""
    from config.loader import settings
    defaults = dict(
        qdrant_host=settings.qdrant.host,
        qdrant_port=settings.qdrant.port,
        collection=settings.qdrant.collection,
        top_k=settings.search.default_top_k,
    )
    defaults.update(kwargs)
    return find_similar(doi, **defaults)


def search_by_author_from_settings(author: str, **kwargs) -> list[SearchResult]:
    """search_by_author() with connection params from config/settings.yaml."""
    from config.loader import settings
    defaults = dict(
        qdrant_host=settings.qdrant.host,
        qdrant_port=settings.qdrant.port,
        collection=settings.qdrant.collection,
    )
    defaults.update(kwargs)
    return search_by_author(author, **defaults)


def get_gap_landscape_from_settings(**kwargs) -> dict:
    """get_gap_landscape() with connection + model params from settings.yaml."""
    from config.loader import settings
    defaults = dict(
        qdrant_host=settings.qdrant.host,
        qdrant_port=settings.qdrant.port,
        collection=settings.qdrant.collection,
        model_name=settings.embedding.model,
        device=settings.embedding.device,
        hybrid_alpha=settings.search.hybrid_alpha,
    )
    defaults.update(kwargs)
    return get_gap_landscape(**defaults)
