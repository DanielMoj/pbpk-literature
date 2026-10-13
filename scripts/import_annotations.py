#!/usr/bin/env python3
"""
Imports annotations saved by export_annotations.py back into Qdrant.

Use after rebuilding the knowledge base (setup_qdrant.py --recreate, then
python -m ingestion.reingest): re-ingesting resets gap_status, tags,
relevant_for and note of every paper. The import matches papers by DOI
(case-insensitive) and sets these four fields on all chunks of each paper
to the exported values. DOIs that are no longer in the collection are listed.

Reads the JSON export (lossless) or the semicolon CSV export.

Usage:
    python scripts/import_annotations.py --input data/annotations_20261013_120000.json --dry-run
    python scripts/import_annotations.py --input data/annotations_20261013_120000.json
    python scripts/import_annotations.py --input data/annotations_20261013_120000.csv
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from config.loader import settings
from retrieval.client import get_client
from retrieval.models import GapStatus

logger = logging.getLogger(__name__)

ANNOTATION_FIELDS = ("gap_status", "tags", "relevant_for", "note")
SCROLL_BATCH = 1000


# ---------------------------------------------------------------------------
# Export file
# ---------------------------------------------------------------------------

def load_annotations(path: Path) -> list[dict]:
    """Read a JSON or CSV export of export_annotations.py into annotation dicts."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        entries = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            entries = list(csv.DictReader(f, delimiter=";"))
        # Listen stehen im CSV kommagetrennt (Tags enthalten keine Kommas)
        for entry in entries:
            for key in ("tags", "relevant_for"):
                entry[key] = [v.strip() for v in (entry.get(key) or "").split(",") if v.strip()]
    else:
        raise ValueError(f"{path.name}: erwartet .json oder .csv aus export_annotations.py")

    if not isinstance(entries, list):
        raise ValueError(f"{path.name}: erwartet eine Liste von Papern")

    valid_statuses = {s.value for s in GapStatus}
    annotations = []
    for nr, entry in enumerate(entries, start=1):
        doi = str(entry.get("doi") or "").strip()
        if not doi:
            raise ValueError(f"{path.name}, Eintrag {nr}: keine DOI")
        status = entry.get("gap_status") or GapStatus.UNASSESSED.value
        if status not in valid_statuses:
            raise ValueError(
                f"{path.name}, Eintrag {nr} ({doi}): unbekannter gap_status '{status}'"
            )
        annotations.append({
            "doi": doi,
            "gap_status": status,
            "tags": list(entry.get("tags") or []),
            "relevant_for": list(entry.get("relevant_for") or []),
            "note": entry.get("note") or "",
        })
    return annotations


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

def doi_index(client: QdrantClient, collection: str) -> dict[str, tuple[str, int]]:
    """Lower-case DOI → (DOI as stored, number of chunks), from one scroll over the collection."""
    index: dict[str, tuple[str, int]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=["doi"],
            with_vectors=False,
        )
        for pt in points:
            doi = (pt.payload or {}).get("doi", "")
            stored, count = index.get(doi.lower(), (doi, 0))
            index[doi.lower()] = (stored, count + 1)
        if offset is None:
            break
    return index


def import_annotations(
    client: QdrantClient,
    collection: str,
    annotations: list[dict],
    dry_run: bool = False,
) -> dict:
    """
    Set the four annotation fields on all chunks of each exported paper.
    Overwrites the current values. Returns counts and the DOIs not found.
    """
    index = doi_index(client, collection)
    papers, chunks, missing = 0, 0, []

    for annotation in annotations:
        hit = index.get(annotation["doi"].lower())
        if hit is None:
            missing.append(annotation["doi"])
            continue
        stored_doi, chunk_count = hit
        if not dry_run:
            client.set_payload(
                collection_name=collection,
                payload={key: annotation[key] for key in ANNOTATION_FIELDS},
                points=Filter(must=[
                    FieldCondition(key="doi", match=MatchValue(value=stored_doi))
                ]),
            )
        papers += 1
        chunks += chunk_count

    return {
        "entries": len(annotations),
        "papers": papers,
        "chunks": chunks,
        "missing": missing,
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import annotations exported by export_annotations.py (matched by DOI)"
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="JSON or CSV file written by export_annotations.py",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only report what would be written",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        annotations = load_annotations(args.input)
    except (OSError, ValueError) as e:
        logger.error(f"Export-Datei nicht lesbar: {e}")
        return 2

    cfg = settings.qdrant
    logger.info(f"Verbinde mit Qdrant @ {cfg.host}:{cfg.port}...")
    try:
        client = get_client(cfg.host, cfg.port)
        exists = client.collection_exists(cfg.collection)
    except Exception as e:
        logger.error(f"Qdrant nicht erreichbar: {e}")
        return 1
    if not exists:
        logger.error(f"Collection '{cfg.collection}' existiert nicht")
        return 1

    summary = import_annotations(client, cfg.collection, annotations, dry_run=args.dry_run)

    label = "DRY RUN — nichts geschrieben: " if args.dry_run else ""
    print(f"\n{label}{summary['papers']} von {summary['entries']} Papern zugeordnet "
          f"({summary['chunks']} Chunks)")
    if summary["missing"]:
        print(f"Nicht in der Collection ({len(summary['missing'])}):")
        for doi in summary["missing"]:
            print(f"  {doi}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
