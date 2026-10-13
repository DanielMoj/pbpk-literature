"""
Tool: get_gap_landscape — topic → all papers with gap_status annotations.

Aggregated overview: papers grouped by gap_status, with summary counts.
Supports optional topic query (hybrid search) and metadata filters.
Primary use: habilitation planning — which gaps are real, which are filled?
"""
import json
import logging
from typing import Annotated

from pydantic import Field

logger = logging.getLogger(__name__)


def get_gap_landscape(
    topic: Annotated[str | None, Field(
        description=(
            "Optional free-text query to scope the landscape, "
            "e.g. 'PBPK validation' or 'VPC misspecification'. "
            "If empty/null, returns all annotated papers (filtered by other params)."
        ),
    )] = None,
    relevant_for: Annotated[str | None, Field(
        description=(
            "Filter by habilitation paper ID, e.g. 'P4', 'B_pm'. "
            "Returns only papers annotated as relevant for that paper."
        ),
    )] = None,
    gap_status: Annotated[str | None, Field(
        description=(
            "Filter by specific gap status. "
            "One of: REAL_GAP, FAST_REAL_GAP, PARTIALLY_FILLED, CROWDED, UNASSESSED. "
            "If not set, excludes UNASSESSED by default."
        ),
    )] = None,
) -> str:
    """Get an aggregated overview of gap annotations, grouped by status, with summary counts."""
    from config.loader import settings
    from retrieval.search import get_gap_landscape as _get_gap_landscape

    try:
        result = _get_gap_landscape(
            query=topic,
            relevant_for=relevant_for,
            gap_status=gap_status,
            qdrant_host=settings.qdrant.host,
            qdrant_port=settings.qdrant.port,
            collection=settings.qdrant.collection,
            model_name=settings.embedding.model,
            device=settings.embedding.device,
            hybrid_alpha=settings.search.hybrid_alpha,
        )
    except Exception as e:
        logger.error(f"get_gap_landscape failed: {e}", exc_info=True)
        return json.dumps({"error": str(e), "groups": {}, "summary": {}})

    return json.dumps(result, ensure_ascii=False, indent=2)
