"""
Tool: search_by_author — author name → metadata filter search.

Pure payload filter on the authors field, no embedding needed.
Returns one result per paper, deduplicated by DOI, sorted by year descending.
For questions like "what has this group published on VPC?"
"""
import json
import logging
from typing import Annotated

from pydantic import Field

logger = logging.getLogger(__name__)


def search_by_author(
    author: Annotated[str, Field(
        description=(
            "Author name to search for: a surname ('Mustermann') or a full name "
            "as written in the paper. Matched as text against the stored "
            "authors list (which may include affiliations)."
        ),
    )],
    top_k: Annotated[int, Field(
        description="Maximum number of papers to return",
    )] = 20,
) -> str:
    """Find all papers by a specific author in the database."""
    from config.loader import settings
    from retrieval.search import search_by_author as _search_by_author

    try:
        results = _search_by_author(
            author=author,
            qdrant_host=settings.qdrant.host,
            qdrant_port=settings.qdrant.port,
            collection=settings.qdrant.collection,
            top_k=top_k,
        )
    except Exception as e:
        logger.error(f"search_by_author failed for '{author}': {e}", exc_info=True)
        return json.dumps({"error": str(e), "results": [], "total": 0})

    output = {
        "author_query": author,
        "results": [_format_result(r) for r in results],
        "total": len(results),
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


def _format_result(r) -> dict:
    ann = r.chunk.metadata.annotation
    return {
        "doi": r.chunk.doi,
        "title": r.chunk.metadata.title,
        "authors": r.chunk.metadata.authors,
        "year": r.chunk.metadata.year,
        "journal": r.chunk.metadata.journal,
        "snippet": r.snippet,
        "gap_status": ann.gap_status.value,
        "tags": ann.tags,
        "relevant_for": ann.relevant_for,
    }
