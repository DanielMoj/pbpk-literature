"""
Tool: get_paper — doi → full extracted content + annotation.

Retrieves all chunks for a given DOI from Qdrant and reconstructs the
document structure: sections in order, plus annotation metadata.
Use after search_papers to read a specific paper in detail.
"""
import json
import logging
from typing import Annotated

from pydantic import Field

logger = logging.getLogger(__name__)


def get_paper(
    doi: Annotated[str, Field(
        description="DOI of the paper to retrieve, e.g. '10.1002/psp4.12345'",
    )],
) -> str:
    """Retrieve full extracted content of a paper by DOI, including all sections and annotations."""
    from config.loader import settings
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    from retrieval.client import get_client
    from retrieval.search import _payload_to_chunk

    try:
        client = get_client(settings.qdrant.host, settings.qdrant.port)

        doi_filter = Filter(must=[
            FieldCondition(key="doi", match=MatchValue(value=doi))
        ])

        points, _ = client.scroll(
            collection_name=settings.qdrant.collection,
            scroll_filter=doi_filter,
            limit=200,  # generous upper bound for sections per paper
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            return json.dumps({"error": f"DOI '{doi}' not found in database"})

        # Reconstruct from chunks — in document order, i.e. by the numeric
        # suffix of chunk_id ("doi___10" after "doi___9", not after "doi___1")
        chunks = [_payload_to_chunk(pt.payload) for pt in points]
        chunks.sort(key=_chunk_position)

        # Metadata is identical across all chunks of the same paper
        meta = chunks[0].metadata
        ann = meta.annotation

        # Build ordered sections list
        sections = []
        for c in chunks:
            sections.append({
                "section_type": c.section_type,
                "text": c.text,
            })

        output = {
            "doi": meta.doi,
            "title": meta.title,
            "authors": meta.authors,
            "year": meta.year,
            "journal": meta.journal,
            "abstract": meta.abstract,
            "annotation": {
                "gap_status": ann.gap_status.value,
                "tags": ann.tags,
                "relevant_for": ann.relevant_for,
                "note": ann.note,
            },
            "sections": sections,
            "total_sections": len(sections),
        }
        return json.dumps(output, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"get_paper failed for DOI {doi}: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


def _chunk_position(chunk) -> int:
    """Section index from chunk_id f"{doi}___{index}" (unknown format → 0)."""
    suffix = chunk.chunk_id.rsplit("___", 1)[-1]
    return int(suffix) if suffix.isdigit() else 0
