#!/usr/bin/env python3
"""
Exports all annotations from Qdrant as JSON and CSV.
Backup + potential sync to Claude Project Knowledge.
scripts/import_annotations.py writes a JSON or CSV export back (by DOI).

Only exports papers with gap_status != UNASSESSED (i.e. actually annotated),
unless --all is specified. For a backup before re-ingesting use --all:
papers with tags or a note but without gap status are otherwise left out.

Usage:
    python scripts/export_annotations.py                       # JSON + CSV
    python scripts/export_annotations.py --format json         # JSON only
    python scripts/export_annotations.py --format csv          # CSV only
    python scripts/export_annotations.py --output /backup/     # Custom output dir
    python scripts/export_annotations.py --all                 # Include UNASSESSED
    python scripts/export_annotations.py --markdown            # Also generate compact MD
"""
import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from config.loader import settings
from retrieval.client import get_client

logger = logging.getLogger(__name__)

SCROLL_BATCH = 100


# ---------------------------------------------------------------------------
# Qdrant data retrieval
# ---------------------------------------------------------------------------

def fetch_annotated_papers(
    client: QdrantClient,
    collection: str,
    include_unassessed: bool = False,
) -> list[dict]:
    """
    Scroll through Qdrant and collect unique annotated papers.
    Returns one entry per DOI with annotation metadata.
    """
    # Build filter: exclude UNASSESSED unless --all
    scroll_filter = None
    if not include_unassessed:
        scroll_filter = Filter(must_not=[
            FieldCondition(key="gap_status", match=MatchValue(value="UNASSESSED"))
        ])

    papers: dict[str, dict] = {}
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for pt in points:
            p = pt.payload
            doi = p.get("doi", "")
            if not doi or doi in papers:
                # Keep existing — prefer abstract section
                if doi in papers:
                    continue
                continue

            papers[doi] = {
                "doi": doi,
                "title": p.get("title", ""),
                "authors": p.get("authors", []),
                "year": p.get("year", 0),
                "journal": p.get("journal", ""),
                "gap_status": p.get("gap_status", "UNASSESSED"),
                "tags": p.get("tags", []),
                "relevant_for": p.get("relevant_for", []),
                "note": p.get("note", ""),
            }

        if offset is None:
            break

    # Sort by year descending, then title
    result = sorted(
        papers.values(),
        key=lambda x: (-x["year"], x["title"]),
    )
    logger.info(f"{len(result)} annotierte Papers gefunden")
    return result


# ---------------------------------------------------------------------------
# Export formats
# ---------------------------------------------------------------------------

def export_json(papers: list[dict], output_path: Path) -> None:
    """Export as formatted JSON."""
    output_path.write_text(
        json.dumps(papers, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"JSON: {output_path} ({len(papers)} Papers)")


def export_csv(papers: list[dict], output_path: Path) -> None:
    """Export as CSV with semicolon separator (tags can contain commas)."""
    fieldnames = [
        "doi", "title", "authors", "year", "journal",
        "gap_status", "tags", "relevant_for", "note",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for paper in papers:
            row = {**paper}
            # Flatten lists to comma-separated strings
            row["authors"] = ", ".join(row["authors"])
            row["tags"] = ", ".join(row["tags"])
            row["relevant_for"] = ", ".join(row["relevant_for"])
            writer.writerow(row)

    logger.info(f"CSV:  {output_path} ({len(papers)} Papers)")


def export_markdown(papers: list[dict], output_path: Path) -> None:
    """
    Export as compact Markdown — suitable for Claude Project Knowledge.
    Groups papers by gap_status, then by relevant_for.
    """
    lines = [
        f"# Annotation Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"**{len(papers)} annotierte Papers**",
        f"",
    ]

    # Group by gap_status
    groups: dict[str, list[dict]] = {}
    for p in papers:
        status = p["gap_status"]
        groups.setdefault(status, []).append(p)

    status_order = ["REAL_GAP", "FAST_REAL_GAP", "PARTIALLY_FILLED", "CROWDED", "UNASSESSED"]
    for status in status_order:
        entries = groups.get(status, [])
        if not entries:
            continue

        lines.append(f"## {status} ({len(entries)})")
        lines.append("")

        for p in entries:
            doi_link = f"https://doi.org/{p['doi']}"
            relevant = ", ".join(p["relevant_for"]) if p["relevant_for"] else "—"
            tags_str = ", ".join(p["tags"]) if p["tags"] else "—"

            lines.append(f"- **{p['title']}** ({p['year']})")
            lines.append(f"  DOI: [{p['doi']}]({doi_link}) | {p['journal']}")
            lines.append(f"  Relevant: {relevant} | Tags: {tags_str}")
            if p["note"]:
                lines.append(f"  Note: {p['note']}")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"MD:   {output_path} ({len(papers)} Papers)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export annotations from Qdrant as JSON/CSV/Markdown"
    )
    parser.add_argument(
        "--format", choices=["json", "csv", "markdown", "all"], default="all",
        help="Export format (default: all)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--all", action="store_true", dest="include_all",
        help="Include UNASSESSED papers (default: only annotated)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = settings.qdrant
    output_dir = args.output or (Path(__file__).resolve().parent.parent / "data")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Verbinde mit Qdrant @ {cfg.host}:{cfg.port}...")
    try:
        client = get_client(cfg.host, cfg.port)
        client.get_collections()
    except Exception as e:
        logger.error(f"Qdrant nicht erreichbar: {e}")
        sys.exit(1)

    papers = fetch_annotated_papers(
        client, cfg.collection, include_unassessed=args.include_all,
    )

    if not papers:
        print("Keine annotierten Papers gefunden.")
        return

    fmt = args.format

    if fmt in ("json", "all"):
        export_json(papers, output_dir / f"annotations_{timestamp}.json")

    if fmt in ("csv", "all"):
        export_csv(papers, output_dir / f"annotations_{timestamp}.csv")

    if fmt in ("markdown", "all"):
        export_markdown(papers, output_dir / f"annotations_{timestamp}.md")

    # Summary
    status_counts = {}
    for p in papers:
        s = p["gap_status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    print(f"\nExport: {len(papers)} Papers")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
