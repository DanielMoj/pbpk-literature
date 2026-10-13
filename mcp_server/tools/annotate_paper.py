"""
Tool: annotate_paper — doi + tags + gap_status + note → Qdrant payload update.

Stores or updates annotations in the Qdrant payload for all chunks of a paper.
Tags and relevant_for are merged additively (existing values preserved).
gap_status and note are overwritten.
"""
import json
import logging
from typing import Annotated

from pydantic import Field

logger = logging.getLogger(__name__)


def annotate_paper(
    doi: Annotated[str, Field(
        description="DOI of the paper to annotate, e.g. '10.1002/psp4.12345'",
    )],
    gap_status: Annotated[str | None, Field(
        description=(
            "Gap status to set. One of: REAL_GAP, FAST_REAL_GAP, "
            "PARTIALLY_FILLED, CROWDED, UNASSESSED. "
            "Overwrites existing value."
        ),
    )] = None,
    tags: Annotated[str | None, Field(
        description=(
            "Comma-separated tags to add, e.g. 'misspecification,NLME,bias'. "
            "Merged with existing tags (additive, duplicates are dropped)."
        ),
    )] = None,
    relevant_for: Annotated[str | None, Field(
        description=(
            "Comma-separated habilitation paper IDs, e.g. 'P4,B_pm,C_pm'. "
            "Merged with existing relevant_for (additive)."
        ),
    )] = None,
    note: Annotated[str | None, Field(
        description=(
            "Free-text note about the paper. Overwrites existing note. "
            "Use for quick assessments, e.g. 'Only 1cmt, no Friberg, no coverage'."
        ),
    )] = None,
) -> str:
    """Add or update annotations (gap status, tags, relevance, notes) for a paper in the database."""
    from config.loader import settings
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    from retrieval.client import get_client
    from retrieval.models import GapStatus

    # Validate gap_status if provided
    valid_statuses = {s.value for s in GapStatus}
    if gap_status and gap_status not in valid_statuses:
        return json.dumps({
            "error": f"Invalid gap_status '{gap_status}'. Must be one of: {sorted(valid_statuses)}",
        })

    # Parse comma-separated inputs
    new_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    new_relevant = [r.strip() for r in relevant_for.split(",") if r.strip()] if relevant_for else []

    # Nothing to do?
    if not gap_status and not new_tags and not new_relevant and note is None:
        return json.dumps({"error": "No annotation fields provided"})

    try:
        client = get_client(settings.qdrant.host, settings.qdrant.port)
        collection = settings.qdrant.collection

        doi_filter = Filter(must=[
            FieldCondition(key="doi", match=MatchValue(value=doi))
        ])

        # Read existing annotation from first chunk
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=doi_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            return json.dumps({"error": f"DOI '{doi}' not found in database"})

        existing = points[0].payload

        # Build updated payload — additive merge for lists, overwrite for scalars
        payload_update = {}

        if gap_status:
            payload_update["gap_status"] = gap_status

        if new_tags:
            existing_tags = existing.get("tags", [])
            merged_tags = list(dict.fromkeys(existing_tags + new_tags))  # deduplicate, preserve order
            payload_update["tags"] = merged_tags

        if new_relevant:
            existing_relevant = existing.get("relevant_for", [])
            merged_relevant = list(dict.fromkeys(existing_relevant + new_relevant))
            payload_update["relevant_for"] = merged_relevant

        if note is not None:
            payload_update["note"] = note

        # Update all chunks for this DOI
        client.set_payload(
            collection_name=collection,
            payload=payload_update,
            points=doi_filter,
        )

        # Build confirmation response
        final_annotation = {
            "gap_status": payload_update.get("gap_status", existing.get("gap_status", "UNASSESSED")),
            "tags": payload_update.get("tags", existing.get("tags", [])),
            "relevant_for": payload_update.get("relevant_for", existing.get("relevant_for", [])),
            "note": payload_update.get("note", existing.get("note", "")),
        }

        return json.dumps({
            "doi": doi,
            "title": existing.get("title", ""),
            "updated_fields": list(payload_update.keys()),
            "annotation": final_annotation,
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"annotate_paper failed for DOI {doi}: {e}", exc_info=True)
        return json.dumps({"error": str(e)})
