"""
Fetches DOI lists from CrossRef and EuropePMC per journal and time range.
Reads journal config from config/journals.yaml.
Output: JSON files in data/doi_lists/ with DOI + basic metadata.

Strategy:
  - Tier 1 (no keywords): CrossRef API — all papers for ISSN + year range
  - Tier 2/3 (keywords):  EuropePMC API — keyword-filtered search

Usage:
    python -m ingestion.fetch_metadata                          # All tiers
    python -m ingestion.fetch_metadata --tier 1                 # Only Tier 1
    python -m ingestion.fetch_metadata --tier 2 --journal "AAPS Journal"
    python -m ingestion.fetch_metadata --year-to 2025
    python -m ingestion.fetch_metadata --dry-run
    python -m ingestion.fetch_metadata --mailto name@example.org
"""
import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import httpx
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CROSSREF_BASE = "https://api.crossref.org/works"
EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

DEFAULT_MAILTO = "pbpk-literature@example.com"
CROSSREF_ROWS = 1000          # items per page (max 1000)
EUROPEPMC_PAGE_SIZE = 100
REQUEST_DELAY = 1.0          # seconds between API requests
HTTP_TIMEOUT = 30.0          # seconds


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class JournalConfig:
    name: str
    issn: str
    year_from: int
    keywords: list[str] | None
    tier: int


@dataclass
class PaperEntry:
    doi: str
    title: str
    authors: list[str]
    journal: str
    year: int
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "doi": self.doi,
            "title": self.title,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "abstract": self.abstract,
            "keywords": self.keywords,
        }


# ---------------------------------------------------------------------------
# Journal config loading
# ---------------------------------------------------------------------------

def load_journal_configs(yaml_path: Path) -> list[JournalConfig]:
    """Load journal definitions from journals.yaml, tagged with tier number."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    configs = []
    for tier_key, tier_num in [("tier_1", 1), ("tier_2", 2), ("tier_3", 3)]:
        entries = raw.get(tier_key, [])
        if not entries:
            continue
        for e in entries:
            configs.append(JournalConfig(
                name=e["name"],
                issn=e["issn"],
                year_from=e.get("year_from", 2010),
                keywords=e.get("keywords"),
                tier=tier_num,
            ))
    return configs


# ---------------------------------------------------------------------------
# Existing DOI loading (dedup)
# ---------------------------------------------------------------------------

def load_existing_dois(doi_lists_dir: Path) -> set[str]:
    """Load all DOIs already fetched from JSON files in data/doi_lists/."""
    existing = set()
    if not doi_lists_dir.exists():
        return existing

    for jf in doi_lists_dir.glob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict) and "doi" in entry:
                        existing.add(entry["doi"].lower())
            elif isinstance(data, dict) and "entries" in data:
                for entry in data["entries"]:
                    if "doi" in entry:
                        existing.add(entry["doi"].lower())
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Überspringe {jf.name}: {e}")

    return existing


# ---------------------------------------------------------------------------
# Filename sanitisation
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """Convert journal name to a safe filename component.

    'CPT: Pharmacometrics & Systems Pharmacology' → 'cpt_pharmacometrics_systems_pharmacology'
    """
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    # Collapse runs of underscores
    name = re.sub(r"_+", "_", name)
    return name


# ---------------------------------------------------------------------------
# CrossRef API (Tier 1 — all papers from a journal)
# ---------------------------------------------------------------------------

def _parse_crossref_author(author: dict) -> str:
    """Parse a CrossRef author dict into 'Lastname F' format."""
    family = author.get("family", "")
    given = author.get("given", "")
    if given:
        initials = "".join(w[0] for w in given.split() if w)
        return f"{family} {initials}"
    return family


def _parse_crossref_item(item: dict, journal_name: str) -> PaperEntry | None:
    """Parse one CrossRef work item into a PaperEntry."""
    doi = item.get("DOI", "")
    if not doi:
        return None

    # Title
    titles = item.get("title", [])
    title = titles[0] if titles else ""

    # Authors
    authors_raw = item.get("author", [])
    authors = [_parse_crossref_author(a) for a in authors_raw]

    # Year — try multiple date fields
    year = 0
    for date_key in ("published", "published-print", "published-online", "created"):
        date_parts = item.get(date_key, {}).get("date-parts", [[]])
        if date_parts and date_parts[0] and date_parts[0][0]:
            year = int(date_parts[0][0])
            break

    # Journal name from response or fallback
    container = item.get("container-title", [])
    journal = container[0] if container else journal_name

    # Abstract (CrossRef sometimes includes JATS XML)
    abstract = item.get("abstract", "")
    if abstract:
        abstract = re.sub(r"<[^>]+>", "", abstract).strip()

    return PaperEntry(
        doi=doi,
        title=title,
        authors=authors,
        journal=journal,
        year=year,
        abstract=abstract,
    )


def fetch_crossref(
    issn: str,
    year_from: int,
    year_to: int,
    mailto: str = DEFAULT_MAILTO,
    delay: float = REQUEST_DELAY,
) -> Iterator[PaperEntry]:
    """
    Fetch all papers from CrossRef for a given ISSN and year range.
    Uses cursor-based pagination. Yields PaperEntry objects.
    """
    cursor = "*"
    total_fetched = 0
    journal_name = ""  # filled from first response

    headers = {
        "User-Agent": f"pbpk-literature/1.0 (mailto:{mailto})",
    }

    with httpx.Client(timeout=HTTP_TIMEOUT, headers=headers) as client:
        while True:
            params = {
                "filter": f"issn:{issn},from-pub-date:{year_from},until-pub-date:{year_to}",
                "rows": CROSSREF_ROWS,
                "cursor": cursor,
                "mailto": mailto,
            }

            try:
                resp = client.get(CROSSREF_BASE, params=params)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                logger.error(f"CrossRef request failed: {e}")
                break

            data = resp.json()
            message = data.get("message", {})
            items = message.get("items", [])

            total_results = message.get("total-results", "?")
            logger.debug(f"  CrossRef page: {len(items)} items, total-results: {total_results}")

            if not items:
                break

            for item in items:
                entry = _parse_crossref_item(item, journal_name)
                if entry:
                    if not journal_name:
                        journal_name = entry.journal
                    yield entry
                    total_fetched += 1

            # Next page
            next_cursor = message.get("next-cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

            logger.debug(f"  CrossRef: {total_fetched} items so far...")
            time.sleep(delay)

    logger.info(f"  CrossRef: {total_fetched} Einträge für ISSN {issn}")


# ---------------------------------------------------------------------------
# EuropePMC API (Tier 2/3 — keyword-filtered)
# ---------------------------------------------------------------------------

def _build_europepmc_query(
    issn: str,
    keywords: list[str],
    year_from: int,
    year_to: int,
) -> str:
    """Build an EuropePMC search query string."""
    kw_parts = " OR ".join(f'"{kw}"' for kw in keywords)
    return (
        f'ISSN:"{issn}" AND ({kw_parts}) '
        f"AND PUB_YEAR:[{year_from} TO {year_to}]"
    )


def _parse_europepmc_author(author_string: str) -> list[str]:
    """Parse EuropePMC authorString into list of 'Lastname F' names."""
    if not author_string:
        return []
    # authorString format: "Mustermann E, Doe AB, Mustermann M."
    authors = []
    for part in author_string.rstrip(".").split(","):
        part = part.strip()
        if part:
            authors.append(part)
    return authors


def _parse_europepmc_item(item: dict) -> PaperEntry | None:
    """Parse one EuropePMC result item into a PaperEntry."""
    doi = item.get("doi", "")
    if not doi:
        return None

    title = item.get("title", "")
    authors = _parse_europepmc_author(item.get("authorString", ""))
    journal = item.get("journalTitle", "")
    abstract = item.get("abstractText", "")

    year = 0
    try:
        year = int(item.get("pubYear", 0))
    except (ValueError, TypeError):
        pass

    # Keywords from EuropePMC
    kw_list = item.get("keywordList", {}).get("keyword", [])

    return PaperEntry(
        doi=doi,
        title=title,
        authors=authors,
        journal=journal,
        year=year,
        abstract=abstract,
        keywords=kw_list if isinstance(kw_list, list) else [],
    )


def fetch_europepmc(
    issn: str,
    keywords: list[str],
    year_from: int,
    year_to: int,
    delay: float = REQUEST_DELAY,
) -> Iterator[PaperEntry]:
    """
    Fetch keyword-filtered papers from EuropePMC.
    Uses cursorMark-based pagination. Yields PaperEntry objects.
    """
    query = _build_europepmc_query(issn, keywords, year_from, year_to)
    cursor_mark = "*"
    total_fetched = 0

    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        while True:
            params = {
                "query": query,
                "format": "json",
                "pageSize": EUROPEPMC_PAGE_SIZE,
                "cursorMark": cursor_mark,
                "resultType": "core",  # includes abstracts
            }

            try:
                resp = client.get(EUROPEPMC_BASE, params=params)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                logger.error(f"EuropePMC request failed: {e}")
                break

            data = resp.json()
            results = data.get("resultList", {}).get("result", [])

            if not results:
                break

            for item in results:
                entry = _parse_europepmc_item(item)
                if entry:
                    yield entry
                    total_fetched += 1

            # Next page
            next_cursor = data.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor_mark:
                break
            cursor_mark = next_cursor

            logger.debug(f"  EuropePMC: {total_fetched} items so far...")
            time.sleep(delay)

    logger.info(f"  EuropePMC: {total_fetched} Einträge für ISSN {issn}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def fetch_journal(
    journal: JournalConfig,
    year_to: int,
    mailto: str,
    existing_dois: set[str],
    delay: float = REQUEST_DELAY,
) -> list[PaperEntry]:
    """
    Fetch DOIs for one journal. Tier 1 uses CrossRef, Tier 2/3 uses EuropePMC.
    Deduplicates against existing_dois.
    """
    logger.info(
        f"Fetching: {journal.name} (Tier {journal.tier}, "
        f"{journal.year_from}–{year_to}, "
        f"{'all' if not journal.keywords else str(len(journal.keywords)) + ' keywords'})"
    )

    if journal.keywords:
        source = fetch_europepmc(
            issn=journal.issn,
            keywords=journal.keywords,
            year_from=journal.year_from,
            year_to=year_to,
            delay=delay,
        )
    else:
        source = fetch_crossref(
            issn=journal.issn,
            year_from=journal.year_from,
            year_to=year_to,
            mailto=mailto,
            delay=delay,
        )

    new_entries = []
    duplicates = 0
    for entry in source:
        if entry.doi.lower() in existing_dois:
            duplicates += 1
            continue
        existing_dois.add(entry.doi.lower())
        new_entries.append(entry)

    logger.info(
        f"  -> {len(new_entries)} neue, {duplicates} bereits vorhanden"
    )
    return new_entries


def save_entries(
    entries: list[PaperEntry],
    output_path: Path,
) -> None:
    """Save PaperEntry list as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [e.to_dict() for e in entries]
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"  Gespeichert: {output_path} ({len(data)} Einträge)")


def fetch_all(
    journals_yaml: Path,
    doi_lists_dir: Path,
    tiers: list[int] | None = None,
    journal_filter: str | None = None,
    year_to: int | None = None,
    mailto: str = DEFAULT_MAILTO,
    delay: float = REQUEST_DELAY,
    dry_run: bool = False,
) -> dict:
    """
    Main entry point: fetch DOIs for all (or filtered) journals.

    Returns summary dict with counts per journal.
    """
    if year_to is None:
        year_to = datetime.now().year

    configs = load_journal_configs(journals_yaml)

    # Apply filters
    if tiers:
        configs = [c for c in configs if c.tier in tiers]
    if journal_filter:
        journal_lower = journal_filter.lower()
        configs = [c for c in configs if journal_lower in c.name.lower()]

    if not configs:
        logger.warning("Keine Journals nach Filterung übrig")
        return {"total_new": 0, "journals": {}}

    logger.info(f"{len(configs)} Journals zu verarbeiten")

    if dry_run:
        for c in configs:
            src = "EuropePMC" if c.keywords else "CrossRef"
            kw_info = f", {len(c.keywords)} keywords" if c.keywords else ""
            print(
                f"  [DRY RUN] {c.name} (Tier {c.tier}, {c.year_from}–{year_to}, "
                f"via {src}{kw_info})"
            )
        return {"total_new": 0, "journals": {}}

    existing_dois = load_existing_dois(doi_lists_dir)
    logger.info(f"{len(existing_dois)} existierende DOIs geladen")

    summary = {}
    total_new = 0

    for journal in configs:
        entries = fetch_journal(
            journal, year_to, mailto, existing_dois, delay
        )

        if entries:
            filename = f"{sanitize_filename(journal.name)}_{journal.year_from}-{year_to}.json"
            output_path = doi_lists_dir / filename

            # Merge with existing file if present
            if output_path.exists():
                try:
                    old_data = json.loads(output_path.read_text(encoding="utf-8"))
                    old_dois = {e["doi"].lower() for e in old_data if "doi" in e}
                    # Only add truly new entries
                    merged = old_data + [
                        e.to_dict() for e in entries
                        if e.doi.lower() not in old_dois
                    ]
                    output_path.write_text(
                        json.dumps(merged, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    logger.info(
                        f"  Merged: {output_path} ({len(old_data)} alt + "
                        f"{len(merged) - len(old_data)} neu = {len(merged)} total)"
                    )
                except (json.JSONDecodeError, KeyError):
                    save_entries(entries, output_path)
            else:
                save_entries(entries, output_path)

        summary[journal.name] = len(entries)
        total_new += len(entries)

    return {"total_new": total_new, "journals": summary}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # Project root detection — same pattern as other scripts
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    parser = argparse.ArgumentParser(
        description="Fetch DOI lists from CrossRef/EuropePMC"
    )
    parser.add_argument(
        "--tier", type=int, nargs="+",
        help="Only fetch specific tiers (1, 2, 3)",
    )
    parser.add_argument(
        "--journal", type=str,
        help="Filter by journal name (substring match)",
    )
    parser.add_argument(
        "--year-to", type=int, default=None,
        help=f"End year (default: {datetime.now().year})",
    )
    parser.add_argument(
        "--mailto", type=str, default=DEFAULT_MAILTO,
        help="Email for CrossRef polite pool (default: %(default)s)",
    )
    parser.add_argument(
        "--delay", type=float, default=REQUEST_DELAY,
        help="Seconds between API requests (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be fetched without making requests",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    journals_yaml = project_root / "config" / "journals.yaml"
    doi_lists_dir = project_root / "data" / "doi_lists"

    if not journals_yaml.exists():
        logger.error(f"journals.yaml nicht gefunden: {journals_yaml}")
        sys.exit(1)

    result = fetch_all(
        journals_yaml=journals_yaml,
        doi_lists_dir=doi_lists_dir,
        tiers=args.tier,
        journal_filter=args.journal,
        year_to=args.year_to,
        mailto=args.mailto,
        delay=args.delay,
        dry_run=args.dry_run,
    )

    print(f"\n{'=' * 50}")
    print(f"  Neue DOIs total: {result['total_new']}")
    for name, count in result.get("journals", {}).items():
        print(f"    {name}: {count}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
