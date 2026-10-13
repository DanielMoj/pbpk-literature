"""
Converts consolidated extraction JSONs into chunks for embedding.
One chunk = one section (Abstract, Methods, Results, Discussion, etc.)
Short sections (<min_chunk_length) get merged with the next section.
Long sections (>max_chunk_length) get split at paragraph boundaries.

Input:  data/extracted/*_extracted.json
Output: List[Chunk] with {text, doi, section_type, authors, journal, year, title}
"""
import json
import logging
import re
from pathlib import Path

from retrieval.models import Chunk, PaperMetadata, Annotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token estimation
# Whitespace-split × subword_factor für wissenschaftliche Texte.
# Formeln, Bindestriche, Abkürzungen (PK/PD, CYP3A4) erzeugen ~30% mehr
# Subword-Tokens als Whitespace-Wörter.
# ---------------------------------------------------------------------------

# Default-Faktor, wird von chunk_document() aus Settings überschrieben
_DEFAULT_SUBWORD_FACTOR = 1.3
_SUBWORD_FACTOR = _DEFAULT_SUBWORD_FACTOR


def _estimate_tokens(text: str, subword_factor: float | None = None) -> int:
    """Rough token count: whitespace words × subword correction factor."""
    factor = subword_factor if subword_factor is not None else _SUBWORD_FACTOR
    return int(len(text.split()) * factor)


# ---------------------------------------------------------------------------
# Section type normalisation
# ---------------------------------------------------------------------------

_SECTION_MAP = {
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "introduction",
    "methods": "methods",
    "materials and methods": "methods",
    "methodology": "methods",
    "experimental": "methods",
    "model": "methods",
    "results": "results",
    "findings": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "summary": "conclusion",
    "references": "references",
    "bibliography": "references",
    "acknowledgements": "acknowledgements",
    "acknowledgments": "acknowledgements",
    "supplementary": "supplementary",
    "supplementary material": "supplementary",
    "supporting information": "supplementary",
    "appendix": "supplementary",
}


def _normalise_section_type(heading: str) -> str:
    """Map a section heading to a canonical section_type."""
    key = re.sub(r"[\d\.\s]+", " ", heading).strip().lower()
    # Try exact match first, then prefix match
    if key in _SECTION_MAP:
        return _SECTION_MAP[key]
    for pattern, stype in _SECTION_MAP.items():
        if key.startswith(pattern):
            return stype
    return "other"


# ---------------------------------------------------------------------------
# Consolidator JSON parsing
# ---------------------------------------------------------------------------

def _extract_metadata(doc: dict) -> PaperMetadata:
    """Extract PaperMetadata from a consolidator JSON.

    Handles two schema variants:
      A) Consolidator v2 (nested):
           doc.metadata.original_metadata  (Metadaten-Sidecar: CrossRef/OpenAlex)
           doc.metadata.document_metadata  (aus Grobid TEI: Title/Authors/Abstract)
      B) Flaches Schema (Legacy/Tests):
           doc.metadata.{doi, title, authors, journal, year, abstract}
    """
    meta_block = doc.get("metadata", doc)

    # --- Variant A: Consolidator v2 ---
    if isinstance(meta_block, dict) and (
        "original_metadata" in meta_block or "document_metadata" in meta_block
    ):
        sidecar = meta_block.get("original_metadata") or {}
        grobid = meta_block.get("document_metadata") or {}

        # DOI/Journal/Year/Publisher aus Sidecar (zuverlässiger aus CrossRef).
        # Title/Abstract aus Grobid (sauberer aus TEI). Authors: Grobid bevorzugt.
        doi = sidecar.get("doi") or grobid.get("doi") or doc.get("document_id", "unknown")
        title = grobid.get("title") or sidecar.get("title", "")
        abstract = grobid.get("abstract") or sidecar.get("abstract", "")
        authors_raw = grobid.get("authors") or sidecar.get("authors") or []
        journal = sidecar.get("journal") or sidecar.get("venue") or grobid.get("journal", "")
        year_raw = sidecar.get("year") or grobid.get("year") or 0

    # --- Variant B: flach ---
    else:
        meta = meta_block if isinstance(meta_block, dict) else {}
        doi = meta.get("doi") or doc.get("doi") or doc.get("id", "unknown")
        title = meta.get("title", "")
        abstract = meta.get("abstract", "")
        authors_raw = meta.get("authors") or []
        journal = meta.get("journal") or meta.get("venue", "")
        year_raw = meta.get("year", 0)

    # Normalise authors: ["Jane Doe", ...] oder [{"name": "Jane Doe"}, ...]
    if authors_raw and isinstance(authors_raw[0], dict):
        authors = [a.get("name", str(a)) for a in authors_raw if a]
    else:
        authors = [str(a) for a in authors_raw if a]

    # Year robust zu int — kann str, None oder fehlend sein
    try:
        year_int = int(year_raw) if year_raw else 0
    except (TypeError, ValueError):
        year_int = 0

    return PaperMetadata(
        doi=doi or "unknown",
        title=str(title or ""),
        authors=authors,
        journal=str(journal or ""),
        year=year_int,
        abstract=str(abstract or ""),
    )


def _extract_raw_sections(doc: dict) -> list[dict]:
    """
    Pull sections from the consolidator JSON.
    Handles multiple schema variants:
      - {"sections": [{"heading": ..., "text": ...}, ...]}
      - {"body": [{"heading": ..., "paragraphs": [...]}]}
      - {"content": {"sections": [...]}}
    """
    # Variant 1: top-level "sections"
    if "sections" in doc and isinstance(doc["sections"], list):
        return doc["sections"]

    # Variant 2: nested under "content"
    content = doc.get("content", {})
    if isinstance(content, dict) and "sections" in content:
        return content["sections"]

    # Variant 3: "body" key
    if "body" in doc and isinstance(doc["body"], list):
        return doc["body"]

    # Variant 4: flat text only
    text = doc.get("text", doc.get("full_text", ""))
    if text:
        return [{"heading": "full_text", "text": text}]

    return []


def _section_text(section: dict) -> str:
    """Get text from a section dict.

    Handles three schema variants:
      - {"text": "..."}
      - {"paragraphs": ["...", ...] | [{"text": "..."}, ...]}
      - Consolidator v2: {"elements": [
            {"type": "paragraph", "text": "..."},
            {"type": "list", "items": [...]},
            ...
        ]}
    """
    # Variant 1: flat text
    if section.get("text"):
        return str(section["text"]).strip()

    # Variant 2: paragraphs list
    if "paragraphs" in section:
        paragraphs = section["paragraphs"]
        if paragraphs and isinstance(paragraphs[0], dict):
            return "\n\n".join(p.get("text", str(p)) for p in paragraphs).strip()
        return "\n\n".join(str(p) for p in paragraphs).strip()

    # Variant 3: Consolidator v2 elements
    if "elements" in section:
        parts: list[str] = []
        for elem in section.get("elements", []):
            if not isinstance(elem, dict):
                continue
            etype = elem.get("type", "")
            if etype == "paragraph":
                text = str(elem.get("text", "")).strip()
                if text:
                    parts.append(text)
            elif etype == "list":
                items = [str(i).strip() for i in elem.get("items", []) if i]
                if items:
                    parts.append("\n".join(f"- {item}" for item in items))
            # Andere Element-Typen (z.B. formulas) ignorieren für Chunking
        return "\n\n".join(parts).strip()

    return ""


def _extract_tables(doc: dict) -> list[dict]:
    """Pull tables from the consolidator JSON."""
    tables = doc.get("tables", [])
    if not tables:
        tables = doc.get("content", {}).get("tables", [])
    return tables


def _table_to_text(tbl: dict) -> str:
    """Render a table dict into embeddable text.

    Handles multiple shapes:
      - Consolidator v2: tbl["content"] = {"rows": [{"cells": [{"text": "..."}, ...]}, ...]}
      - Legacy: tbl["content"] as string
      - Legacy: tbl["text"] as string
      - Fallback: empty if nothing usable
    """
    content = tbl.get("content")

    # Nothing to render
    if content is None:
        return str(tbl.get("text", "")).strip()

    # Flat string (legacy)
    if isinstance(content, str):
        return content.strip()

    # Consolidator v2: dict with "rows"
    if isinstance(content, dict) and "rows" in content:
        lines: list[str] = []
        for row in content.get("rows", []):
            if not isinstance(row, dict):
                continue
            cells = row.get("cells", [])
            cell_texts = []
            for cell in cells:
                if isinstance(cell, dict):
                    cell_texts.append(str(cell.get("text", "")).strip())
                else:
                    cell_texts.append(str(cell).strip())
            if any(cell_texts):
                lines.append(" | ".join(cell_texts))
        return "\n".join(lines).strip()

    # Unknown shape — try str() as last resort but strip dict noise
    return ""


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def _accumulate_splits(segments: list[str], max_tokens: int, joiner: str) -> list[str]:
    """Greedy-accumulate segments into chunks that stay under max_tokens."""
    chunks = []
    current: list[str] = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = _estimate_tokens(seg)
        if current_tokens + seg_tokens > max_tokens and current:
            chunks.append(joiner.join(current))
            current = [seg]
            current_tokens = seg_tokens
        else:
            current.append(seg)
            current_tokens += seg_tokens

    if current:
        chunks.append(joiner.join(current))

    return chunks


def _hard_split_words(text: str, max_tokens: int) -> list[str]:
    """Last resort: split at word boundaries to stay under max_tokens.
    Uses raw word count (no subword factor) to guarantee the split works."""
    words = text.split()
    # Convert max_tokens back to approximate word count
    max_words = max(10, int(max_tokens / _SUBWORD_FACTOR))
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks


def _split_at_paragraphs(text: str, max_tokens: int) -> list[str]:
    """Split text to stay under max_tokens.

    Strategy (cascading):
    1. Split at paragraph boundaries (\\n\\n)
    2. If any chunk still too long: split at sentence boundaries (. )
    3. If still too long: hard split at word boundaries
    """
    # Level 1: Paragraph split
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = _accumulate_splits(paragraphs, max_tokens, "\n\n")

    # Level 2: Sentence split for oversized chunks
    final = []
    for chunk in chunks:
        if _estimate_tokens(chunk) > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            sub_chunks = _accumulate_splits(sentences, max_tokens, " ")
            # Level 3: Hard word split for any remaining oversized parts
            for sc in sub_chunks:
                if _estimate_tokens(sc) > max_tokens:
                    final.extend(_hard_split_words(sc, max_tokens))
                else:
                    final.append(sc)
        else:
            final.append(chunk)

    return final


def chunk_document(
    doc: dict,
    min_chunk_length: int = 50,
    max_chunk_length: int = 350,
    subword_factor: float | None = None,
) -> list[Chunk]:
    """
    Convert a single consolidator JSON into a list of Chunks.

    Strategy:
    1. Parse sections from the JSON
    2. Merge short sections with the next section
    3. Split long sections at paragraph boundaries
    4. Add tables as separate chunks
    5. Prepend abstract as first chunk if available

    Args:
        subword_factor: Override module default for token estimation.
                        If None, uses _DEFAULT_SUBWORD_FACTOR (1.3).
    """
    global _SUBWORD_FACTOR
    # None heißt Standardfaktor, nicht der Faktor eines früheren Aufrufs
    _SUBWORD_FACTOR = (
        subword_factor if subword_factor is not None else _DEFAULT_SUBWORD_FACTOR
    )

    metadata = _extract_metadata(doc)
    raw_sections = _extract_raw_sections(doc)

    if not raw_sections and not metadata.abstract:
        logger.warning(f"No content found for DOI {metadata.doi}")
        return []

    # --- Build section list with heading + text ---
    sections: list[tuple[str, str]] = []

    # Abstract als erste Sektion (falls nicht schon in sections)
    if metadata.abstract:
        has_abstract = any(
            _normalise_section_type(s.get("heading", "")) == "abstract"
            for s in raw_sections
        )
        if not has_abstract:
            sections.append(("abstract", metadata.abstract))

    for raw in raw_sections:
        heading = raw.get("heading", raw.get("title", "other"))
        text = _section_text(raw)
        if not text:
            continue
        stype = _normalise_section_type(heading)
        if stype == "references":
            continue  # Referenzlisten nicht embedden
        sections.append((stype, text))

    # --- Merge short sections ---
    # Abstract und supplementary nie mergen — semantisch eigenständig
    _NO_MERGE_TYPES = {"abstract", "supplementary"}

    merged: list[tuple[str, str]] = []
    i = 0
    while i < len(sections):
        stype, text = sections[i]
        tokens = _estimate_tokens(text)

        should_merge = (
            tokens < min_chunk_length
            and stype not in _NO_MERGE_TYPES
        )

        # Kurze Sektion: mit nächster mergen (forward)
        if should_merge and i + 1 < len(sections):
            next_stype, next_text = sections[i + 1]
            # Nächste Sektion auch geschützt? Dann nicht mergen.
            if next_stype in _NO_MERGE_TYPES:
                merged.append((stype, text))
                i += 1
            else:
                combined_text = text + "\n\n" + next_text
                merged.append((next_stype, combined_text))
                i += 2
        # Kurze letzte Sektion: rückwärts an vorherige anhängen
        elif should_merge and merged:
            prev_stype, prev_text = merged[-1]
            merged[-1] = (prev_stype, prev_text + "\n\n" + text)
            i += 1
        else:
            merged.append((stype, text))
            i += 1

    # --- Split long sections ---
    final_sections: list[tuple[str, str]] = []
    for stype, text in merged:
        tokens = _estimate_tokens(text)
        if tokens > max_chunk_length:
            parts = _split_at_paragraphs(text, max_chunk_length)
            for part in parts:
                final_sections.append((stype, part))
        else:
            final_sections.append((stype, text))

    # --- Add tables as chunks ---
    tables = _extract_tables(doc)
    for tbl in tables:
        caption = str(tbl.get("caption", "")).strip()
        tbl_body = _table_to_text(tbl)
        if not tbl_body:
            continue
        tbl_text = f"{caption}\n\n{tbl_body}".strip() if caption else tbl_body
        if _estimate_tokens(tbl_text) >= min_chunk_length:
            final_sections.append(("table", tbl_text))

    # --- Build Chunk objects ---
    chunks = []
    for idx, (stype, text) in enumerate(final_sections):
        chunk_id = f"{metadata.doi}___{idx}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            doi=metadata.doi,
            section_type=stype,
            text=text,
            metadata=metadata,
        ))

    if not chunks:
        logger.warning(f"No chunks produced for DOI {metadata.doi}")

    return chunks


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def chunk_directory(
    input_dir: Path,
    min_chunk_length: int = 50,
    max_chunk_length: int = 350,
    subword_factor: float | None = None,
) -> list[Chunk]:
    """
    Process all *_extracted.json files in a directory.
    Returns flat list of all chunks.

    Args:
        subword_factor: Override module default for token estimation.
                        Passed through to chunk_document().
    """
    json_files = sorted(input_dir.glob("*_extracted.json"))
    if not json_files:
        logger.warning(
            f"Keine *_extracted.json in {input_dir} gefunden. "
            f"Consolidator-Pipeline zuerst ausführen."
        )
        return []

    all_chunks: list[Chunk] = []
    errors = 0

    for jf in json_files:
        try:
            doc = json.loads(jf.read_text(encoding="utf-8"))
            chunks = chunk_document(
                doc, min_chunk_length, max_chunk_length, subword_factor
            )
            all_chunks.extend(chunks)
            logger.debug(f"{jf.name}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Fehler bei {jf.name}: {e}")
            errors += 1

    logger.info(
        f"Chunking fertig: {len(json_files)} Files -> {len(all_chunks)} Chunks "
        f"({errors} Fehler)"
    )
    return all_chunks
