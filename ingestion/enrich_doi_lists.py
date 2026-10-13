"""
Enriches existing DOI list JSONs with abstracts and keywords.
Backfills missing data from EuropePMC (primary) and Semantic Scholar (fallback).

Idempotent: skips entries that already have an abstract.
Standalone module.

Usage:
    python -m ingestion.enrich_doi_lists                              # All lists
    python -m ingestion.enrich_doi_lists --input data/doi_lists/specific_file.json
    python -m ingestion.enrich_doi_lists --dry-run                    # Count missing only
    python -m ingestion.enrich_doi_lists --source europepmc           # Skip S2 fallback
"""
import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
S2_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"

EUROPEPMC_DELAY = 0.15   # ~7 req/s, well under their limit
S2_DELAY = 1.1           # 1 req/s without API key (conservative)
HTTP_TIMEOUT = 20.0


# ---------------------------------------------------------------------------
# EuropePMC
# ---------------------------------------------------------------------------

def _fetch_europepmc(doi: str, client: httpx.Client) -> dict | None:
    """
    Fetch abstract + keywords from EuropePMC for a single DOI.
    Returns {"abstract": str, "keywords": list[str]} or None.
    """
    params = {
        "query": f'DOI:"{doi}"',
        "format": "json",
        "resultType": "core",
        "pageSize": 1,
    }
    try:
        resp = client.get(EUROPEPMC_SEARCH, params=params)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("resultList", {}).get("result", [])
        if not results:
            return None

        item = results[0]
        abstract = (item.get("abstractText") or "").strip()
        keywords = item.get("keywordList", {}).get("keyword", [])
        if not isinstance(keywords, list):
            keywords = []

        if not abstract:
            return None

        return {"abstract": abstract, "keywords": keywords}

    except Exception as e:
        logger.debug(f"EuropePMC failed for {doi}: {e}")
        return None


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

def _fetch_semantic_scholar(doi: str, client: httpx.Client) -> dict | None:
    """
    Fetch abstract + tldr from Semantic Scholar for a single DOI.
    Returns {"abstract": str, "tldr": str, "keywords": []} or None.
    """
    url = S2_PAPER_URL.format(doi=doi)
    params = {"fields": "abstract,tldr"}
    try:
        resp = client.get(url, params=params)
        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            logger.warning("S2 rate limit hit — backing off 5s")
            time.sleep(5)
            resp = client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        abstract = (data.get("abstract") or "").strip()
        tldr_obj = data.get("tldr")
        tldr = tldr_obj.get("text", "").strip() if isinstance(tldr_obj, dict) else ""

        if not abstract and not tldr:
            return None

        result = {"abstract": abstract, "keywords": []}
        if tldr:
            result["tldr"] = tldr
        return result

    except Exception as e:
        logger.debug(f"S2 failed for {doi}: {e}")
        return None


# ---------------------------------------------------------------------------
# Core enrichment
# ---------------------------------------------------------------------------

@dataclass
class EnrichStats:
    total: int = 0
    already_have: int = 0
    enriched_epmc: int = 0
    enriched_s2: int = 0
    not_found: int = 0


def enrich_entries(
    entries: list[dict],
    client: httpx.Client,
    use_s2: bool = True,
    dry_run: bool = False,
) -> EnrichStats:
    """
    Enrich a list of DOI entries in-place.
    Modifies entries directly — caller handles file I/O.
    """
    stats = EnrichStats(total=len(entries))

    missing = [e for e in entries if not (e.get("abstract") or "").strip()]
    stats.already_have = stats.total - len(missing)

    if dry_run:
        stats.not_found = len(missing)
        return stats

    for i, entry in enumerate(missing):
        doi = entry.get("doi", "")
        if not doi:
            stats.not_found += 1
            continue

        # --- EuropePMC first ---
        time.sleep(EUROPEPMC_DELAY)
        result = _fetch_europepmc(doi, client)

        if result:
            entry["abstract"] = result["abstract"]
            if result["keywords"] and not entry.get("keywords"):
                entry["keywords"] = result["keywords"]
            stats.enriched_epmc += 1
            _log_progress(i, len(missing), doi, "EuropePMC")
            continue

        # --- S2 fallback ---
        if use_s2:
            time.sleep(S2_DELAY)
            result = _fetch_semantic_scholar(doi, client)

            if result:
                entry["abstract"] = result["abstract"]
                if result.get("tldr"):
                    entry["tldr"] = result["tldr"]
                stats.enriched_s2 += 1
                _log_progress(i, len(missing), doi, "S2")
                continue

        stats.not_found += 1
        _log_progress(i, len(missing), doi, "—")

    return stats


def _log_progress(i: int, total: int, doi: str, source: str):
    if (i + 1) % 25 == 0 or i + 1 == total:
        logger.info(f"  [{i+1}/{total}] last: {doi} via {source}")


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

def process_file(
    json_path: Path,
    client: httpx.Client,
    use_s2: bool = True,
    dry_run: bool = False,
) -> EnrichStats:
    """Load a DOI list JSON, enrich, write back."""
    entries = json.loads(json_path.read_text(encoding="utf-8"))

    if not isinstance(entries, list):
        # Handle {"entries": [...]} wrapper
        if isinstance(entries, dict) and "entries" in entries:
            inner = entries["entries"]
        else:
            logger.warning(f"Unexpected format in {json_path.name} — skip")
            return EnrichStats()
    else:
        inner = entries

    logger.info(f"{json_path.name}: {len(inner)} Einträge")
    stats = enrich_entries(inner, client, use_s2=use_s2, dry_run=dry_run)

    if not dry_run and (stats.enriched_epmc + stats.enriched_s2) > 0:
        # Write back (entries was modified in-place)
        json_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"  -> {json_path.name} aktualisiert")

    return stats


def process_directory(
    doi_lists_dir: Path,
    use_s2: bool = True,
    dry_run: bool = False,
) -> EnrichStats:
    """Process all JSON files in a directory."""
    json_files = sorted(doi_lists_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"Keine JSON-Dateien in {doi_lists_dir}")
        return EnrichStats()

    totals = EnrichStats()
    client = httpx.Client(timeout=HTTP_TIMEOUT)

    try:
        for jf in json_files:
            s = process_file(jf, client, use_s2=use_s2, dry_run=dry_run)
            totals.total += s.total
            totals.already_have += s.already_have
            totals.enriched_epmc += s.enriched_epmc
            totals.enriched_s2 += s.enriched_s2
            totals.not_found += s.not_found
    finally:
        client.close()

    return totals


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    parser = argparse.ArgumentParser(
        description="Enrich DOI lists with abstracts from EuropePMC + Semantic Scholar"
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=None,
        help="Single JSON file or directory (default: data/doi_lists/)",
    )
    parser.add_argument(
        "--source", choices=["all", "europepmc", "s2"], default="all",
        help="Which sources to use (default: all = EuropePMC -> S2 fallback)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only count missing abstracts, don't fetch",
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

    target = args.input or (project_root / "data" / "doi_lists")
    use_s2 = args.source in ("all", "s2")

    if target.is_file():
        client = httpx.Client(timeout=HTTP_TIMEOUT)
        try:
            stats = process_file(target, client, use_s2=use_s2, dry_run=args.dry_run)
        finally:
            client.close()
    elif target.is_dir():
        stats = process_directory(target, use_s2=use_s2, dry_run=args.dry_run)
    else:
        logger.error(f"Nicht gefunden: {target}")
        sys.exit(1)

    label = "DRY RUN — " if args.dry_run else ""
    print(f"\n{'=' * 50}")
    print(f"  {label}Enrichment Ergebnis")
    print(f"  Total:            {stats.total}")
    print(f"  Abstract vorhanden: {stats.already_have}")
    print(f"  Neu via EuropePMC: {stats.enriched_epmc}")
    print(f"  Neu via S2:        {stats.enriched_s2}")
    print(f"  Nicht gefunden:    {stats.not_found}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
