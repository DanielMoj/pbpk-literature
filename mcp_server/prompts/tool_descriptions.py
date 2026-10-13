"""
Clean tool descriptions for Claude.
These are registered via @server.tool() decorators in the individual tool modules.

This file serves as documentation and reference — the actual descriptions
are defined as docstrings and Field annotations in each tool function.
"""

TOOL_DESCRIPTIONS = {
    "search_papers": (
        "Hybrid semantic + keyword search over the PBPK literature database. "
        "Combines dense embedding similarity with BM25 lexical matching via "
        "weighted Reciprocal Rank Fusion (weight hybrid_alpha). Supports metadata "
        "filters on journal, year range, annotation tags, gap status, and "
        "habilitation paper relevance."
    ),
    "get_paper": (
        "Retrieve the full extracted content of a paper by DOI. Returns all "
        "sections (abstract, methods, results, discussion, etc.) plus current "
        "annotations (gap status, tags, relevance, notes). Use after search_papers "
        "to read a specific paper in detail."
    ),
    "annotate_paper": (
        "Add or update annotations for a paper. Gap status and notes are overwritten; "
        "tags and relevant_for lists are merged additively. Updates all chunks of the "
        "paper atomically. Use during literature review to build the gap landscape."
    ),
    "find_similar": (
        "Find papers most similar to a given paper based on embedding proximity. "
        "Uses the dense vector of a specified section (default: abstract) as the "
        "query. Useful for finding related work and identifying research clusters."
    ),
    "search_by_author": (
        "Find all papers by a specific author. Pure metadata filter search (text "
        "match on the authors list), no embedding needed. Returns one result per "
        "paper, deduplicated by DOI, sorted by year descending."
    ),
    "get_gap_landscape": (
        "Aggregated overview of gap annotations. Papers grouped by gap status "
        "(REAL_GAP, FAST_REAL_GAP, PARTIALLY_FILLED, CROWDED) with summary counts. "
        "Supports scoping by topic query or habilitation paper ID. Excludes "
        "UNASSESSED papers by default."
    ),
}
