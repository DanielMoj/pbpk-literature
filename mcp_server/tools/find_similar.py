"""
Tool: find_similar — doi → N nearest neighbors in embedding space.

Retrieves the dense vector of a paper's section (default: abstract) and
searches for the most similar papers. Useful for questions like
"which papers do something similar to this simulation study?"
"""
import json
import logging
from typing import Annotated

from pydantic import Field

logger = logging.getLogger(__name__)


def find_similar(
    doi: Annotated[str, Field(
        description="DOI of the anchor paper, e.g. '10.1002/psp4.12345'",
    )],
    section: Annotated[str, Field(
        description=(
            "Which section to use as embedding anchor. "
            "One of: abstract, methods, results, discussion, conclusion. "
            "Defaults to 'abstract' — usually the best summary vector."
        ),
    )] = "abstract",
    top_k: Annotated[int, Field(
        description="Number of similar papers to return",
    )] = 10,
) -> str:
    """Find papers most similar to a given paper based on embedding proximity."""
    from config.loader import settings
    from retrieval.search import find_similar as _find_similar

    try:
        results = _find_similar(
            doi=doi,
            section=section,
            qdrant_host=settings.qdrant.host,
            qdrant_port=settings.qdrant.port,
            collection=settings.qdrant.collection,
            top_k=top_k,
        )
    except Exception as e:
        logger.error(f"find_similar failed for DOI {doi}: {e}", exc_info=True)
        return json.dumps({"error": str(e), "results": [], "total": 0})

    output = {
        "anchor_doi": doi,
        "anchor_section": section,
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
        "section_type": r.chunk.section_type,
        "snippet": r.snippet,
        "score": round(r.score, 4),
        "gap_status": ann.gap_status.value,
        "tags": ann.tags,
        "relevant_for": ann.relevant_for,
    }
