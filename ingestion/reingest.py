"""
Convenience wrapper: takes a directory of consolidated JSONs,
runs chunk → embed → store in sequence.
For bulk import and incremental updates.

Usage:
    python -m ingestion.reingest --input data/extracted/
    python -m ingestion.reingest --input data/extracted/ --device cpu
"""
import argparse
import logging
import sys
from pathlib import Path

# Projekt-Root zum sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import settings
from ingestion.chunk_papers import chunk_directory
from ingestion.embed_and_store import embed_and_store

logger = logging.getLogger(__name__)


def reingest(
    input_dir: Path,
    device: str | None = None,
    delete_before_insert: bool = True,
):
    """
    Full pipeline: JSONs → Chunks → Embeddings → Qdrant.

    Args:
        input_dir: Directory containing *_extracted.json files
        device: Override device from settings ('auto', 'cuda' or 'cpu')
        delete_before_insert: Delete existing DOI data before re-ingestion
    """
    cfg = settings

    # Step 1: Chunk
    logger.info(f"Chunking JSONs aus {input_dir}...")
    chunks = chunk_directory(
        input_dir,
        min_chunk_length=cfg.chunking.min_chunk_length,
        max_chunk_length=cfg.chunking.max_chunk_length,
        subword_factor=cfg.chunking.subword_factor,
    )

    if not chunks:
        logger.warning("Keine Chunks erzeugt — Abbruch.")
        return

    # Step 2: Embed + Store
    logger.info(f"Ingestiere {len(chunks)} Chunks in Qdrant...")
    embed_and_store(
        chunks=chunks,
        qdrant_host=cfg.qdrant.host,
        qdrant_port=cfg.qdrant.port,
        collection=cfg.qdrant.collection,
        model_name=cfg.embedding.model,
        device=device or cfg.embedding.device,
        batch_size=cfg.embedding.batch_size,
        delete_before_insert=delete_before_insert,
    )

    # Summary
    dois = {c.doi for c in chunks}
    logger.info(f"Fertig: {len(dois)} Papers, {len(chunks)} Chunks ingestiert.")


def main():
    parser = argparse.ArgumentParser(
        description="Reingest consolidated JSONs into Qdrant"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Directory with *_extracted.json files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override embedding device (auto/cuda/cpu)",
    )
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Don't delete existing chunks before re-ingestion",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input.exists():
        logger.error(f"Input-Verzeichnis existiert nicht: {args.input}")
        sys.exit(1)

    reingest(
        input_dir=args.input,
        device=args.device,
        delete_before_insert=not args.no_delete,
    )


if __name__ == "__main__":
    main()
