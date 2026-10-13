"""
Tool: search_papers — query + filters → top_k results.

Semantic + lexical hybrid search over PBPK/pharmacometrics papers.
Combines dense embedding (bge-large-en-v1.5) with BM25 sparse vectors
via weighted Reciprocal Rank Fusion (weight hybrid_alpha from
config/settings.yaml). Supports metadata filters on journal, year, tags,
gap_status, and relevance to specific habilitation papers.

The search itself is retrieval.search.search_from_settings; eval/trefferquote.py
calls the same function.
"""
import json
import logging
from typing import Annotated

from pydantic import Field

logger = logging.getLogger(__name__)


def search_papers(
    query: Annotated[str, Field(
        description=(
            "Free-text search query. Supports natural language and exact terms "
            "(e.g. 'PBPK validation TOST' or 'VPC power misspecification'). "
            "Can be empty string if only metadata filters are used."
        ),
    )],
    journal: Annotated[str | None, Field(
        description="Filter by exact journal name, e.g. 'CPT: Pharmacometrics & Systems Pharmacology'",
    )] = None,
    year_min: Annotated[int | None, Field(
        description="Include only papers published in this year or later",
    )] = None,
    year_max: Annotated[int | None, Field(
        description="Include only papers published in this year or earlier",
    )] = None,
    tags: Annotated[str | None, Field(
        description=(
            "Comma-separated annotation tags to filter on. "
            "All tags must match (AND logic). "
            "Examples: 'misspecification', 'TOST,DDI', 'VPC,power'"
        ),
    )] = None,
    gap_status: Annotated[str | None, Field(
        description=(
            "Filter by gap status annotation. "
            "One of: REAL_GAP, FAST_REAL_GAP, PARTIALLY_FILLED, CROWDED, UNASSESSED"
        ),
    )] = None,
    relevant_for: Annotated[str | None, Field(
        description=(
            "Filter by habilitation paper ID, e.g. 'P4', 'B_pm', 'NP3'. "
            "Returns only papers annotated as relevant for that paper."
        ),
    )] = None,
    top_k: Annotated[int, Field(
        description="Maximum number of results to return",
    )] = 10,
) -> str:
    """Search the PBPK literature database with semantic + keyword hybrid search and metadata filters."""
    from retrieval.search import search_from_settings

    # Parse comma-separated tags into list
    tag_list = None
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    try:
        results = search_from_settings(
            query,
            top_k=top_k,
            journal=journal,
            year_min=year_min,
            year_max=year_max,
            tags=tag_list,
            gap_status=gap_status,
            relevant_for=relevant_for,
        )
    except Exception as e:
        logger.error(f"search_papers failed: {e}", exc_info=True)
        return json.dumps({"error": str(e), "results": [], "total": 0})

    output = {
        "results": [_format_result(r) for r in results],
        "total": len(results),
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


def _format_result(r) -> dict:
    """Serialize a SearchResult into a JSON-friendly dict."""
    ann = r.chunk.metadata.annotation
    return {
        "doi": r.chunk.doi,
        "title": r.chunk.metadata.title,
        "authors": r.chunk.metadata.authors,
        "year": r.chunk.metadata.year,
        "journal": r.chunk.metadata.journal,
        "section_type": r.chunk.section_type,
        "snippet": r.snippet,
        "score": round(r.score, 4),
        "gap_status": ann.gap_status.value,
        "tags": ann.tags,
        "relevant_for": ann.relevant_for,
    }
