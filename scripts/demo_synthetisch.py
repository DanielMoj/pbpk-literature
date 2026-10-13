#!/usr/bin/env python3
"""
Demo ohne Docker und ohne eigene PDFs: das synthetische Testkorpus
(tests/fixtures/korpus, vier erfundene Paper) wird mit den echten Modellen
(bge-large-en-v1.5, BM25) in Qdrant im Speicher eingespielt, dann wird das
MCP-Werkzeug search_papers aufgerufen und seine Antwort ausgegeben.

Der erste Lauf lädt das Embedding-Modell (~1,3 GB) und das BM25-Modell von
Hugging Face. Gerechnet wird nach embedding.device (auto: GPU, wenn vorhanden).

Usage:
    python scripts/demo_synthetisch.py
    python scripts/demo_synthetisch.py --frage "Which interval decides bioequivalence?" --top-k 2
"""
import argparse
import logging
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from qdrant_client import QdrantClient  # noqa: E402

from config.loader import settings  # noqa: E402
from ingestion.chunk_papers import chunk_directory  # noqa: E402
from ingestion.embed_and_store import embed_and_store_from_settings  # noqa: E402
from mcp_server.tools.search_papers import search_papers  # noqa: E402
from retrieval.client import use_client  # noqa: E402
from setup_qdrant import create_collection  # noqa: E402

KORPUS = PROJECT_ROOT / "tests" / "fixtures" / "korpus"


def main():
    parser = argparse.ArgumentParser(description="search_papers am synthetischen Korpus")
    parser.add_argument("--frage", default="How does renal impairment change the exposure of Fiktivol?")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Modellabfragen bei Hugging Face

    cfg = settings.chunking
    chunks = chunk_directory(
        KORPUS,
        min_chunk_length=cfg.min_chunk_length,
        max_chunk_length=cfg.max_chunk_length,
        subword_factor=cfg.subword_factor,
    )

    client = QdrantClient(":memory:")
    create_collection(client, settings.qdrant.collection)
    use_client(client)
    embed_and_store_from_settings(chunks)

    print(f"\nsearch_papers(query={args.frage!r}, top_k={args.top_k})\n", flush=True)
    print(search_papers(args.frage, top_k=args.top_k))


if __name__ == "__main__":
    main()
